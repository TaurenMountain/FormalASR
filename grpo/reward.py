"""
grpo/reward.py
奖励函数：CER 奖励、Qwen-Judge 裁判奖励，以及混合奖励构建器。
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import requests

from common.metrics import compute_cer


# ──────────────────────────────────────────────────────────────────────
# CER 奖励
# ──────────────────────────────────────────────────────────────────────

def cer_reward(hyp: str, ref: str) -> float:
    """
    将 CER 转换为 [-1, 1] 奖励：
      cer=0  → reward=+1（完全正确）
      cer=2  → reward=-1（clip 下界）
      hyp 为空 → reward=-1（惩罚空输出）
    """
    if not (hyp or "").strip():
        return -1.0
    cer = compute_cer(ref, hyp)
    return float(max(-1.0, min(1.0, 1.0 - min(cer, 2.0))))


# ──────────────────────────────────────────────────────────────────────
# Qwen-Judge 裁判奖励
# ──────────────────────────────────────────────────────────────────────

class QwenJudgeScorer:
    """通过 Qwen 大模型接口对 (hypothesis, target) 打分，返回 [-1, 1]。"""

    SYSTEM_PROMPT = (
        "你是一名专业的语音识别（ASR）转写质量评估员。\n"
        "你的任务是评估微调 ASR 模型的转写输出（hypothesis）与正确文本（target）的字符级吻合程度。\n\n"
        "评分规则（0~10 整数）：\n"
        "  10 ── 逐字完全一致，或仅有可忽略的全半角/空格差异\n"
        "   9 ── 仅 1 个同音字/近音字替换，语义无损\n"
        "   7 ── 2~3 个字符错误（替换/插入/删除），关键信息完整\n"
        "   5 ── 字符错误率约 10%~20%，主体语义可理解但有明显缺失\n"
        "   3 ── 字符错误率约 20%~50%，语义严重失真\n"
        "   1 ── 字符错误率 > 50%，或关键词全部错误\n"
        "   0 ── hypothesis 为空，或与 target 毫无关联\n\n"
        "注意：\n"
        "- 以字符级准确率为主要依据，不要依赖语义猜测\n"
        "- 数字、专名、单位的错误权重高于虚词\n"
        "- 忽略标点符号差异，不因标点扣分\n\n"
        "只输出一个 0~10 的整数，不要输出任何其他内容。"
    )

    def __init__(
        self,
        url: str,
        model: str = "qwen35-27b",
        timeout: float = 20.0,
        fail_value: float = 0.0,
    ):
        self.url = url
        self.model = model
        self.timeout = timeout
        self.fail_value = float(max(-1.0, min(1.0, fail_value)))

    def _call_one(self, hyp: str, target: str) -> float:
        payload = {
            "model": self.model,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"target: {target}\n"
                        f"hypothesis: {hyp}"
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 8,
        }
        try:
            resp = requests.post(
                self.url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return self.fail_value
            data = resp.json()
            text = (
                (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
                .strip()
            )
            m = re.search(r"-?\d+(?:\.\d+)?", text)
            if not m:
                return self.fail_value
            score_0_10 = max(0.0, min(10.0, float(m.group())))
            return (score_0_10 / 5.0) - 1.0  # 映射到 [-1, 1]
        except Exception:
            return self.fail_value

    def score_batch(
        self,
        hyps: List[str],
        refs: List[str],
        max_workers: int = 8,
    ) -> List[float]:
        """并发调用 API 对整批样本打分，避免串行等待拖慢训练。"""
        n = len(hyps)
        results: List[float] = [self.fail_value] * n
        with ThreadPoolExecutor(max_workers=min(max_workers, n or 1)) as pool:
            futures = {
                pool.submit(self._call_one, h, r): i
                for i, (h, r) in enumerate(zip(hyps, refs))
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception:
                    results[idx] = self.fail_value
        return results


# ──────────────────────────────────────────────────────────────────────
# 混合奖励构建器
# ──────────────────────────────────────────────────────────────────────

def build_reward_fn(
    cer_weight: float = 1.0,
    judge_weight: float = 0.0,
    judge_scorer: Optional[QwenJudgeScorer] = None,
):
    """
    构建混合奖励函数（加权平均），结果天然在 [-1, 1]：
      total = (cer_weight * cer_score + judge_weight * judge_score)
              / (cer_weight + judge_weight)

    cer_score  ∈ [-1, 1]（由 cer_reward 保证）
    judge_score ∈ [-1, 1]（由 QwenJudgeScorer._call_one 保证）

    返回的 reward_fn(hyps, refs, return_parts=False)：
      - return_parts=False → List[float]（总奖励，∈ [-1, 1]）
      - return_parts=True  → (总奖励, cer_scores, judge_scores) 的三元组
    """
    _cer_w   = max(cer_weight,   0.0)
    _judge_w = max(judge_weight, 0.0)
    _total_w = _cer_w + _judge_w
    if _total_w <= 0:
        _total_w = 1.0  # 防止除零，退化为全零奖励

    def reward_fn(hyps: List[str], refs: List[str], return_parts: bool = False):
        # cer_reward 已保证返回 [-1, 1]
        cer_scores = [cer_reward(h, r) for h, r in zip(hyps, refs)]

        # judge_scorer._call_one 已保证返回 [-1, 1]
        judge_scores = [0.0] * len(cer_scores)
        if _judge_w > 0 and judge_scorer is not None:
            judge_scores = judge_scorer.score_batch(hyps, refs)

        # 加权平均：分子是加权和，分母是权重之和 → 结果 ∈ [-1, 1]
        totals = [
            float((_cer_w * c + _judge_w * j) / _total_w)
            for c, j in zip(cer_scores, judge_scores)
        ]

        if return_parts:
            return totals, cer_scores, judge_scores
        return totals

    return reward_fn

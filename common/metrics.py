"""
common/metrics.py
文本指标计算：CER、WER、ROUGE-L、BERTScore、S²ER，以及文本规范化工具。
被 grpo/reward.py 和 eval/run.py 共同使用。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

# 路径常量（供 BERTScorer 查找本地模型）
_THIS_FILE    = Path(__file__).resolve()
_COMMON_DIR   = _THIS_FILE.parent          # common/
_FINETUNE_DIR = _COMMON_DIR.parent         # asr-finetune/
_PROJECT_ROOT = _FINETUNE_DIR.parent       # nwy/


# ──────────────────────────────────────────────────────────────────────
# 文本规范化
# ──────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    规范化文本用于评估：
      - 去除首尾空格
      - 全角转半角数字/字母
      - 压缩多余空格
    """
    text = (text or "").strip()
    text = text.translate(str.maketrans(
        "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
        "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    ))
    return re.sub(r"\s+", " ", text)


# ──────────────────────────────────────────────────────────────────────
# 编辑距离（字符级 / 词级通用）
# ──────────────────────────────────────────────────────────────────────

def edit_distance(s1, s2) -> int:
    """计算两个序列之间的编辑距离（Levenshtein），支持字符串或列表。"""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ──────────────────────────────────────────────────────────────────────
# CER / WER
# ──────────────────────────────────────────────────────────────────────

def compute_cer(ref: str, hyp: str) -> float:
    """字符错误率（CER）= 字符级编辑距离 / 参考长度。"""
    ref = normalize_text(ref)
    hyp = normalize_text(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def compute_wer(ref: str, hyp: str) -> float:
    """词错误率（WER）= 词级编辑距离 / 参考词数。"""
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


# ──────────────────────────────────────────────────────────────────────
# ROUGE-L
# ──────────────────────────────────────────────────────────────────────

def compute_rouge_l(ref: str, hyp: str) -> float:
    """计算 ROUGE-L（基于最长公共子序列，字符级）。"""
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    m, n = len(ref), len(hyp)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[n]
    precision = lcs / n
    recall    = lcs / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ──────────────────────────────────────────────────────────────────────
# BERTScore
# ──────────────────────────────────────────────────────────────────────

_bert_scorer = None


def get_bert_scorer(device: str = "cpu", model_type: str = "bert-base-chinese"):
    """懒加载 BERTScorer。

    优先使用本地 bert-base-chinese 模型（PROJECT_ROOT/models/bert-base-chinese/）；
    若本地路径不存在则回退到 HuggingFace hub 下载。
    """
    global _bert_scorer
    if _bert_scorer is None:
        try:
            from bert_score import BERTScorer
            local_path = str(_PROJECT_ROOT / "models" / "bert-base-chinese")
            if Path(local_path).is_dir():
                effective_model = local_path
                num_layers = 12   # bert-base-chinese 共 12 层
            else:
                effective_model = model_type
                num_layers = None
            kwargs = dict(
                model_type=effective_model,
                rescale_with_baseline=False,
                device=device,
            )
            if num_layers is not None:
                kwargs["num_layers"] = num_layers
            _bert_scorer = BERTScorer(**kwargs)
            print(f"[指标] BERTScorer 已初始化 model={effective_model}（device={device}）✓")
        except ImportError:
            print("[警告] bert_score 未安装，BERTScore 将全部为 0.0。请运行: pip install bert-score")
            _bert_scorer = None
    return _bert_scorer


def compute_bertscore_batch(refs: List[str], hyps: List[str], device: str = "cpu") -> List[float]:
    """批量计算 BERTScore F1，返回每条的 F1 分数列表。"""
    scorer = get_bert_scorer(device)
    if scorer is None:
        return [0.0] * len(refs)
    try:
        _, _, F1 = scorer.score(hyps, refs)
        return F1.tolist()
    except Exception as e:
        print(f"[警告] BERTScore 计算失败: {e}")
        return [0.0] * len(refs)


# ──────────────────────────────────────────────────────────────────────
# S²ER（Semantic Sentence Error Rate）
# ──────────────────────────────────────────────────────────────────────

_S2ER_SYSTEM_PROMPT = (
    "你是一名语音识别质量评估专家。"
    "你的任务是判断两段文本在语义上是否等价——即两者表达的核心意图是否相同，"
    "即使用词存在细微差异也算等价。\n"
    "请只回答 1（语义等价）或 0（语义不等价），不要输出任何其他内容。"
)

_S2ER_USER_TEMPLATE = "参考文本：{ref}\nASR识别结果：{hyp}\n是否语义等价（1/0）："

# 默认本地 Qwen-27B 服务地址（与 grpo/train.py 的 reward_judge_url 保持一致）
_DEFAULT_S2ER_URL   = "http://124.70.98.246:9500/v1/chat/completions"
_DEFAULT_S2ER_MODEL = "qwen35-27b"


def _judge_one(
    ref: str,
    hyp: str,
    url: str,
    model: str,
    timeout: float = 30.0,
) -> Optional[int]:
    """向 Qwen-27B（或其他 OpenAI 兼容服务）发一次请求。

    返回值：
      1    — 语义等价
      0    — 语义不等价
      None — 网络失败 / HTTP 非 200 / 超时 / 返回内容无法解析，计算时跳过该条
    """
    import requests as _requests

    if not ref and not hyp:
        return 1

    payload = {
        "model": model,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": _S2ER_SYSTEM_PROMPT},
            {"role": "user",   "content": _S2ER_USER_TEMPLATE.format(ref=ref, hyp=hyp)},
        ],
        "temperature": 0.0,
        "max_tokens": 4,
    }
    try:
        resp = _requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            print(f"[警告] S²ER judge HTTP {resp.status_code}（ref={ref[:20]!r}），跳过该条")
            return None
        text = (
            (((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content") or "")
            .strip()
        )
        digit = next((c for c in text if c in "01"), None)
        if digit is None:
            print(f"[警告] S²ER judge 返回无效内容 {text!r}（ref={ref[:20]!r}），跳过该条")
            return None
        return int(digit)
    except Exception as e:
        print(f"[警告] S²ER judge 调用失败（ref={ref[:20]!r}）: {e}，跳过该条")
        return None


def _llm_judge_semantic(
    refs: List[str],
    hyps: List[str],
    api_base: str = _DEFAULT_S2ER_URL,
    model: str = _DEFAULT_S2ER_MODEL,
    api_key: str = "EMPTY",        # 保留参数签名兼容性，实际不使用
    timeout: float = 30.0,
    max_workers: int = 8,
) -> List[Optional[int]]:
    """
    并发调用本地 Qwen-27B 批量判断语义等价性。
    与 grpo/reward.py 的 QwenJudgeScorer.score_batch 采用相同的并发方式。

    参数：
      api_base   : 完整 chat/completions URL，
                   例如 http://124.70.98.246:9500/v1/chat/completions
      model      : 模型名称，例如 qwen35-27b
      max_workers: 并发线程数

    返回每条判断结果列表（1=等价，0=不等价，None=网络失败/超时跳过）。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    n = len(refs)
    results: List[Optional[int]] = [None] * n

    with ThreadPoolExecutor(max_workers=min(max_workers, n or 1)) as pool:
        futures = {
            pool.submit(_judge_one, r, h, api_base, model, timeout): i
            for i, (r, h) in enumerate(zip(refs, hyps))
        }
        for fut in _as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"[警告] S²ER future 异常（idx={idx}）: {e}，跳过该条")
                results[idx] = None

    skipped = sum(1 for r in results if r is None)
    if skipped:
        print(f"[S²ER] 共 {skipped}/{n} 条因失败被跳过，不计入最终统计")

    return results


def compute_s2er_batch(
    refs: List[str],
    hyps: List[str],
    api_base: str = _DEFAULT_S2ER_URL,
    model: str = _DEFAULT_S2ER_MODEL,
    api_key: str = "EMPTY",
    timeout: float = 30.0,
    max_workers: int = 8,
) -> float:
    """
    计算 S²ER（Semantic Sentence Error Rate）。

    S²ER = (1/N') Σ (1 - LLM_judge(Y_i, Y_GT_i))
         = 1 - (语义等价句子数 / 有效判断句子数)

    网络失败/超时的条目返回 None，**不计入分母**，避免系统性高估错误率。
    若所有条目均失败，返回 None。

    使用本地 Qwen-27B 作为 LLM judge，调用方式与 grpo/reward.py 的
    QwenJudgeScorer 保持一致（requests + chat_template_kwargs）。

    参数：
      refs        : 参考文本列表
      hyps        : ASR 识别结果列表
      api_base    : 完整 chat/completions URL
      model       : 模型名称
      timeout     : 单次请求超时秒数
      max_workers : 并发线程数

    返回：
      s2er : float，范围 [0, 1]，越低越好；若全部失败则返回 None
    """
    n = len(refs)
    if n == 0:
        return 0.0
    judgments = _llm_judge_semantic(
        refs, hyps,
        api_base=api_base,
        model=model,
        api_key=api_key,
        timeout=timeout,
        max_workers=max_workers,
    )
    # 过滤掉 None（网络失败/超时），只统计有效判断
    valid = [j for j in judgments if j is not None]
    if not valid:
        print("[警告] S²ER 所有条目均失败，无法计算，返回 None")
        return None
    num_equivalent = sum(valid)
    return 1.0 - num_equivalent / len(valid)

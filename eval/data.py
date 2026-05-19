"""
eval/data.py
评估数据加载：从 JSONL 加载评估记录，从已有结果文件重算指标。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from common.metrics import (
    compute_cer,
    compute_wer,
    compute_rouge_l,
    compute_bertscore_batch,
    compute_s2er_batch,
    normalize_text,
)


# ──────────────────────────────────────────────────────────────────────
# 加载评估 JSONL
# ──────────────────────────────────────────────────────────────────────

def load_eval_records(jsonl_path: str, max_samples: int = -1) -> List[Dict]:
    """
    从评估标注 JSONL 加载记录，每条包含：
      idx, split, target_text, original_text, asr_text, language
    """
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            idx          = rec.get("idx")
            target_text  = rec.get("target_text", "")
            if idx is None or not target_text:
                continue

            records.append({
                "idx":           int(idx),
                "split":         rec.get("split", "test"),
                "target_text":   target_text,
                "original_text": rec.get("original_text", ""),
                "asr_text":      rec.get("asr_text", ""),
                "language":      rec.get("language", "Chinese"),
                "audio_path":    rec.get("audio_path", ""),
            })
            if max_samples > 0 and len(records) >= max_samples:
                break

    return records


# ──────────────────────────────────────────────────────────────────────
# 从已有结果文件直接重算指标（跳过推理）
# ──────────────────────────────────────────────────────────────────────

def compute_metrics_from_file(
    eval_jsonl: str,
    model_tag: str = "model",
    bert_device: str = "cpu",
    s2er_api_base: Optional[str] = None,
    s2er_model: str = "default",
    s2er_api_key: str = "EMPTY",
) -> Dict[str, float]:
    """
    从已保存的评估 JSONL 文件中直接汇总指标，无需重新推理。

    JSONL 每行必须包含 cer / wer / exact 字段（由 eval/run.py 的 evaluate() 写入）。
    若缺少 rouge_l / bertscore_f1 / s2er 字段则现场重算。

    参数：
      s2er_api_base : 若提供，则计算 S²ER（需要 OpenAI 兼容 API）；为 None 时跳过。
    """
    total_cer     = 0.0
    total_wer     = 0.0
    total_rouge_l = 0.0
    total_bert_f1 = 0.0
    total_s2er    = 0.0
    exact_matches = 0
    n = 0
    recs_for_bert:  List[Dict] = []
    recs_for_s2er:  List[Dict] = []

    with open(eval_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            ref = normalize_text(rec.get("target", rec.get("target_text", "")))
            hyp = normalize_text(rec.get("hypothesis", ""))

            if "cer" in rec and "wer" in rec and "exact" in rec:
                total_cer += float(rec["cer"])
                total_wer += float(rec["wer"])
                exact_matches += int(bool(rec["exact"]))
            else:
                total_cer += compute_cer(ref, hyp)
                total_wer += compute_wer(ref, hyp)
                exact_matches += int(ref == hyp)

            if "rouge_l" in rec:
                total_rouge_l += float(rec["rouge_l"])
            else:
                total_rouge_l += compute_rouge_l(ref, hyp)

            if "bertscore_f1" in rec:
                total_bert_f1 += float(rec["bertscore_f1"])
            else:
                recs_for_bert.append({"ref": ref, "hyp": hyp, "idx": n})

            # S²ER：优先读取已存字段，否则收集待计算
            if "s2er" in rec:
                total_s2er += float(rec["s2er"])
            elif s2er_api_base is not None:
                recs_for_s2er.append({"ref": ref, "hyp": hyp, "idx": n})

            n += 1

    if n == 0:
        raise RuntimeError(f"评估结果文件为空或无法解析：{eval_jsonl}")

    if recs_for_bert:
        print(f"\n[指标] 文件中缺少 bertscore_f1，现场批量计算 {len(recs_for_bert)} 条...")
        refs_b = [r["ref"] for r in recs_for_bert]
        hyps_b = [r["hyp"] for r in recs_for_bert]
        f1s    = compute_bertscore_batch(refs_b, hyps_b, device=bert_device)
        total_bert_f1 += sum(f1s)

    s2er_value: Optional[float] = None
    if recs_for_s2er and s2er_api_base is not None:
        print(f"\n[指标] 文件中缺少 s2er，现场批量计算 {len(recs_for_s2er)} 条...")
        refs_s = [r["ref"] for r in recs_for_s2er]
        hyps_s = [r["hyp"] for r in recs_for_s2er]
        partial_s2er = compute_s2er_batch(refs_s, hyps_s,
                                          api_base=s2er_api_base,
                                          model=s2er_model,
                                          api_key=s2er_api_key)
        # partial_s2er 是平均值，还原总和再加上已存总和后重新平均
        total_s2er += partial_s2er * len(recs_for_s2er)
        s2er_value = round(total_s2er / n, 4)
    elif total_s2er > 0:
        # 全部来自已存字段
        s2er_value = round(total_s2er / n, 4)

    result: Dict = {
        "model_tag":        model_tag,
        "num_samples":      n,
        "avg_cer":          round(total_cer    / n, 4),
        "avg_wer":          round(total_wer    / n, 4),
        "exact_match":      round(exact_matches / n, 4),
        "avg_rouge_l":      round(total_rouge_l / n, 4),
        "avg_bertscore_f1": round(total_bert_f1 / n, 4),
    }
    if s2er_value is not None:
        result["s2er"] = s2er_value
    return result

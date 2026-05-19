"""
eval/run.py
评估主循环：从 HuggingFace Hub 加载数据集，推理、计算指标、结果写入与汇总。

运行示例（单数据集）：
    CUDA_VISIBLE_DEVICES=0 python -m eval.run \\
        --checkpoint TaurenMountain/FormalASR-1.7B \\
        --dataset_names wenetspeech \\
        --output_dir eval_results/ \\
        --batch_size 4

运行示例（多数据集）：
    CUDA_VISIBLE_DEVICES=0 python -m eval.run \\
        --checkpoint TaurenMountain/FormalASR-1.7B \\
        --dataset_names wenetspeech,speechio \\
        --output_dir eval_results/ \\
        --batch_size 4 --max_samples 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from common.audio import (
    TARGET_SR,
    decode_audio_from_hf,
    load_hf_dataset,
)
from common.metrics import (
    compute_bertscore_batch,
    compute_cer,
    compute_rouge_l,
    compute_s2er_batch,
    compute_wer,
    normalize_text,
)
from .data import compute_metrics_from_file
from .infer import load_asr_model, transcribe_batch

_THIS_DIR = Path(__file__).resolve().parent   # eval/


# ──────────────────────────────────────────────────────────────────────
# 从 HF Dataset 构建评估记录
# ──────────────────────────────────────────────────────────────────────

def build_eval_records(
    hf_dataset,
    dataset_name: str,
    split: str = "test",
    max_samples: int = -1,
) -> List[Dict]:
    """
    从 HF Dataset 直接构建评估记录列表。
    每条记录包含：idx、split、target_text、original_text、language、dataset_name
    """
    records = []
    n = len(hf_dataset)
    if max_samples > 0:
        n = min(n, max_samples)

    for i in range(n):
        row = hf_dataset[i]
        target = row.get("target_text", "")
        if not target:
            continue
        records.append({
            "idx":           i,
            "split":         row.get("split", split),
            "target_text":   target,
            "original_text": row.get("original_text", ""),
            "language":      row.get("language", "Chinese"),
            "dataset_name":  dataset_name,
        })

    return records


# ──────────────────────────────────────────────────────────────────────
# 主评估流程
# ──────────────────────────────────────────────────────────────────────

def evaluate(
    model,
    records: List[Dict],
    hf_datasets: Dict[str, Any],
    output_file: Optional[str] = None,
    model_tag: str = "model",
    force_language: bool = True,
    batch_size: int = 1,
    bert_device: str = "cpu",
    s2er_api_base: Optional[str] = None,
    s2er_model: str = "default",
    s2er_api_key: str = "EMPTY",
    resume: bool = True,
) -> Dict[str, float]:
    """
    对给定样本列表进行推理并计算 CER / WER / EM / ROUGE-L / BERTScore。
    音频从 hf_datasets[dataset_name:split] 按 idx 读取。

    resume=True 时，若 output_file 已存在则跳过已完成的样本续写。
    """
    total_cer     = 0.0
    total_wer     = 0.0
    total_rouge_l = 0.0
    exact_matches = 0
    all_refs: List[str] = []
    all_hyps: List[str] = []
    results: List[Dict] = []

    # ── Resume：读取已完成的记录 ─────────────────────────────────────
    done_keys: set = set()
    if resume and output_file and Path(output_file).is_file():
        with open(output_file, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _rec = json.loads(_line)
                    _ds  = _rec.get("dataset_name", "")
                    _idx = _rec.get("idx")
                    if _idx is not None:
                        done_keys.add((_ds, int(_idx)))
                    _ref = normalize_text(_rec.get("target", _rec.get("target_text", "")))
                    _hyp = normalize_text(_rec.get("hypothesis", ""))
                    total_cer     += float(_rec.get("cer",     compute_cer(_ref, _hyp)))
                    total_wer     += float(_rec.get("wer",     compute_wer(_ref, _hyp)))
                    total_rouge_l += float(_rec.get("rouge_l", compute_rouge_l(_ref, _hyp)))
                    exact_matches += int(bool(_rec.get("exact", _ref == _hyp)))
                    all_refs.append(_ref)
                    all_hyps.append(_hyp)
                    results.append(_rec)
                except Exception:
                    pass
        if done_keys:
            print(f"[Resume] 从断点恢复：已完成 {len(done_keys)} 条，续推剩余样本。")

    pending_records = [
        r for r in records
        if (r.get("dataset_name", ""), int(r.get("idx", -1))) not in done_keys
    ] if done_keys else records

    write_mode = "a" if (resume and done_keys) else "w"
    fout = open(output_file, write_mode, encoding="utf-8") if output_file else None

    def chunked(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i: i + n]

    pbar = tqdm(
        list(chunked(pending_records, batch_size)),
        desc=f"评估[{model_tag}]",
        unit=f"batch(x{batch_size})",
    )

    for batch in pbar:
        audio_arrays: List[np.ndarray] = []
        for rec in batch:
            idx     = rec.get("idx")
            split   = rec.get("split", "test")
            ds_name = rec.get("dataset_name", "")
            try:
                composite = f"{ds_name}:{split}" if ds_name else split
                ds = hf_datasets.get(composite) or hf_datasets.get(split)
                if ds is None:
                    raise KeyError(
                        f"hf_datasets 中找不到键 {composite!r}，"
                        f"可用键: {list(hf_datasets.keys())}"
                    )
                row = ds[int(idx)]
                arr = decode_audio_from_hf(row["audio"])
            except Exception as e:
                print(f"\n[警告] 读取音频失败 idx={idx} ds={ds_name}: {e}")
                arr = np.zeros(TARGET_SR, dtype=np.float32)
            audio_arrays.append(arr)

        languages = [rec["language"] for rec in batch]
        hyps = transcribe_batch(model, audio_arrays, languages, force_language)

        for rec, hyp in zip(batch, hyps):
            hyp_norm = normalize_text(hyp)
            ref_norm = normalize_text(rec["target_text"])

            cer     = compute_cer(ref_norm, hyp_norm)
            wer     = compute_wer(ref_norm, hyp_norm)
            rouge_l = compute_rouge_l(ref_norm, hyp_norm)
            exact   = (hyp_norm == ref_norm)

            total_cer     += cer
            total_wer     += wer
            total_rouge_l += rouge_l
            exact_matches += int(exact)
            all_refs.append(ref_norm)
            all_hyps.append(hyp_norm)

            result_rec = {
                "idx":           rec.get("idx"),
                "split":         rec.get("split", "test"),
                "dataset_name":  rec.get("dataset_name", ""),
                "language":      rec["language"],
                "original_text": rec.get("original_text", ""),
                "target":        rec["target_text"],
                "hypothesis":    hyp,
                "cer":           round(cer,     4),
                "wer":           round(wer,     4),
                "rouge_l":       round(rouge_l, 4),
                "exact":         exact,
            }
            results.append(result_rec)

            if fout:
                fout.write(json.dumps(result_rec, ensure_ascii=False) + "\n")
                fout.flush()

        n = len(results)
        pbar.set_postfix({
            "avg_CER":     f"{total_cer/n:.3f}",
            "avg_ROUGE-L": f"{total_rouge_l/n:.3f}",
        })

    if fout:
        fout.close()

    # 批量计算 BERTScore
    print("\n[指标] 计算 BERTScore（全量批次）...")
    bert_f1_list  = compute_bertscore_batch(all_refs, all_hyps, device=bert_device)
    total_bert_f1 = sum(bert_f1_list)

    # 批量计算 S²ER（可选）
    s2er_list: List[Optional[float]] = [None] * len(all_refs)
    avg_s2er: Optional[float] = None
    if s2er_api_base:
        print("\n[指标] 计算 S²ER（全量批次，调用 LLM judge）...")
        from common.metrics import _llm_judge_semantic
        judgments = _llm_judge_semantic(
            all_refs, all_hyps,
            api_base=s2er_api_base,
            model=s2er_model,
            api_key=s2er_api_key,
        )
        s2er_list = [1 - j for j in judgments]
        n_total = max(len(judgments), 1)
        avg_s2er = round(1.0 - sum(judgments) / n_total, 4)

    if output_file and Path(output_file).is_file():
        lines = Path(output_file).read_text(encoding="utf-8").splitlines()
        with open(output_file, "w", encoding="utf-8") as fw:
            for idx_line, (line, bf1) in enumerate(zip(lines, bert_f1_list)):
                try:
                    rec_out = json.loads(line)
                    rec_out["bertscore_f1"] = round(float(bf1), 4)
                    if s2er_list[idx_line] is not None:
                        rec_out["s2er"] = float(s2er_list[idx_line])
                    fw.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                except Exception:
                    fw.write(line + "\n")

    n = max(len(results), 1)
    summary: Dict[str, Any] = {
        "model_tag":        model_tag,
        "num_samples":      len(results),
        "avg_cer":          round(total_cer     / n, 4),
        "avg_wer":          round(total_wer     / n, 4),
        "exact_match":      round(exact_matches / n, 4),
        "avg_rouge_l":      round(total_rouge_l / n, 4),
        "avg_bertscore_f1": round(total_bert_f1 / n, 4),
    }
    if avg_s2er is not None:
        summary["s2er"] = avg_s2er
    return summary


# ──────────────────────────────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("FormalASR 评估")

    p.add_argument("--checkpoint",    type=str, required=True,
                   help="微调后的 checkpoint 目录或 HF 模型 ID（必填）；"
                        "也支持 pipeline / openai/<model> / 本地 .gguf 文件路径")
    p.add_argument("--dataset_names", type=str, default="wenetspeech",
                   help="数据集名称，多个用逗号分隔（别名或完整 HF repo ID）。"
                        "例如：wenetspeech,speechio")
    p.add_argument("--hf_cache_dir",  type=str, default=None,
                   help="HuggingFace 数据集缓存目录")
    p.add_argument("--output_dir",    type=str,
                   default=str(_THIS_DIR.parent / "eval_results"),
                   help="评估结果输出目录")
    p.add_argument("--max_samples",   type=int, default=-1,
                   help="每个数据集最多评估多少条（-1=全部）")
    p.add_argument("--device",        type=str, default="cuda:0")
    p.add_argument("--quantization",  type=str, default="none",
                   choices=["none", "8bit", "4bit"],
                   help="推理量化方式：none / 8bit / 4bit")
    p.add_argument("--infer_force_language", type=int, default=1,
                   help="推理时是否传 language（1=传语言，0=language=None）")
    p.add_argument("--batch_size",    type=int, default=1)
    p.add_argument("--bert_device",   type=str, default="cpu",
                   help="BERTScore 计算设备（默认 cpu；可设为 cuda:0 加速）")
    p.add_argument("--s2er_api_base", type=str, default=None,
                   help="S²ER LLM judge URL，例如 http://localhost:9500/v1/chat/completions。"
                        "不传则跳过 S²ER 计算。")
    p.add_argument("--s2er_model",    type=str, default="qwen35-27b")
    p.add_argument("--s2er_api_key",  type=str, default="EMPTY")
    p.add_argument("--resume",        type=int, default=1,
                   help="1=从断点恢复，0=覆盖重来")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# 辅助打印
# ──────────────────────────────────────────────────────────────────────

def _print_summary(s: Dict, indent: str = "  ") -> None:
    print(f"{indent}模型:              {s['model_tag']}")
    print(f"{indent}样本数:            {s['num_samples']}")
    print(f"{indent}平均 CER:          {s['avg_cer']:.4f}  ({s['avg_cer']*100:.2f}%)")
    print(f"{indent}平均 WER:          {s['avg_wer']:.4f}  ({s['avg_wer']*100:.2f}%)")
    print(f"{indent}精确匹配率:        {s['exact_match']:.4f}  ({s['exact_match']*100:.2f}%)")
    print(f"{indent}平均 ROUGE-L:      {s['avg_rouge_l']:.4f}  ({s['avg_rouge_l']*100:.2f}%)")
    print(f"{indent}平均 BERTScore F1: {s['avg_bertscore_f1']:.4f}  ({s['avg_bertscore_f1']*100:.2f}%)")
    if "s2er" in s:
        print(f"{indent}S²ER:              {s['s2er']:.4f}  ({s['s2er']*100:.2f}%)")


# ──────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    dataset_names: List[str] = [n.strip() for n in args.dataset_names.split(",") if n.strip()]
    if not dataset_names:
        raise ValueError("--dataset_names 不能为空")

    ckpt_name  = Path(args.checkpoint).name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  FormalASR 评估（支持多数据集）")
    print(f"  checkpoint:   {args.checkpoint}")
    print(f"  数据集列表:   {dataset_names}")
    print(f"  输出目录:     {output_dir}")
    print(f"  设备:         {args.device}")
    print(f"  量化:         {args.quantization}")
    print(f"  batch_size:   {args.batch_size}")
    print(f"  resume:       {bool(args.resume)}")
    print("=" * 60)

    # ── 1. 从 HF Hub 加载各数据集评估 split ────────────────────────
    print("\n[数据] 从 HuggingFace Hub 加载各数据集评估 split...")
    all_records: List[Dict] = []
    hf_datasets: Dict[str, Any] = {}

    for ds_name in dataset_names:
        # 根据数据集决定评估 split
        eval_split = "test"
        try:
            print(f"  [{ds_name}] 尝试加载 test split...")
            ds = load_hf_dataset(ds_name, "test", cache_dir=args.hf_cache_dir)
        except Exception:
            try:
                print(f"  [{ds_name}] 未找到 test，尝试 validation split...")
                ds = load_hf_dataset(ds_name, "validation", cache_dir=args.hf_cache_dir)
                eval_split = "validation"
            except Exception as e:
                raise RuntimeError(
                    f"未能从 HuggingFace 加载数据集 [{ds_name}] 的评估 split: {e}"
                )

        key = f"{ds_name}:{eval_split}"
        hf_datasets[key] = ds

        recs = build_eval_records(
            ds, dataset_name=ds_name, split=eval_split,
            max_samples=args.max_samples,
        )
        all_records.extend(recs)
        print(f"    → {len(recs)} 条样本")

    print(f"\n  评估样本总数: {len(all_records)}")
    if not all_records:
        raise RuntimeError("所有数据集的评估数据均为空")

    # ── 2. 确定输出文件名 ──────────────────────────────────────────
    ft_output = str(output_dir / f"{ckpt_name}_eval.jsonl")

    # ── 3. 检查是否所有样本均已完成（resume 模式） ─────────────────
    _do_infer = True
    if args.resume and Path(ft_output).is_file():
        _done_count = 0
        with open(ft_output, "r", encoding="utf-8") as _f:
            for _l in _f:
                if _l.strip():
                    try:
                        json.loads(_l)
                        _done_count += 1
                    except Exception:
                        pass
        if _done_count >= len(all_records):
            print(f"\n[跳过推理] 所有 {len(all_records)} 条样本均已完成，直接计算指标。")
            _do_infer = False

    # ── 4. 加载模型并推理 ─────────────────────────────────────────
    per_ds_summaries: Dict[str, Dict] = {}
    if _do_infer:
        print(f"\n[推理] 评估模型: {args.checkpoint}  batch_size={args.batch_size}")
        ft_model = load_asr_model(
            args.checkpoint,
            args.device,
            batch_size=args.batch_size,
            quantization=args.quantization,
        )
        ft_summary = evaluate(
            ft_model,
            all_records,
            hf_datasets,
            output_file=ft_output,
            model_tag=ckpt_name,
            force_language=bool(args.infer_force_language),
            batch_size=args.batch_size,
            bert_device=args.bert_device,
            s2er_api_base=args.s2er_api_base,
            s2er_model=args.s2er_model,
            s2er_api_key=args.s2er_api_key,
            resume=bool(args.resume),
        )
        del ft_model
        torch.cuda.empty_cache()
    else:
        ft_summary = compute_metrics_from_file(
            ft_output, model_tag=ckpt_name, bert_device=args.bert_device,
            s2er_api_base=args.s2er_api_base,
            s2er_model=args.s2er_model,
            s2er_api_key=args.s2er_api_key,
        )

    # ── 5. 按数据集拆分结果，分别汇总 ─────────────────────────────
    if len(dataset_names) > 1:
        _lines_by_ds: Dict[str, List[str]] = {n: [] for n in dataset_names}
        with open(ft_output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    _ds = rec.get("dataset_name", "")
                    if _ds in _lines_by_ds:
                        _lines_by_ds[_ds].append(line)
                except Exception:
                    pass
        for ds_name, lines in _lines_by_ds.items():
            if not lines:
                continue
            per_ds_file = str(output_dir / f"{ckpt_name}_{ds_name}_eval.jsonl")
            with open(per_ds_file, "w", encoding="utf-8") as fw:
                fw.write("\n".join(lines) + "\n")
            per_ds_summaries[ds_name] = compute_metrics_from_file(
                per_ds_file, model_tag=f"{ckpt_name}_{ds_name}",
                bert_device=args.bert_device,
                s2er_api_base=args.s2er_api_base,
                s2er_model=args.s2er_model,
                s2er_api_key=args.s2er_api_key,
            )

    # ── 6. 打印汇总 ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  评估结果汇总（所有数据集合并）")
    print("=" * 60)
    _print_summary(ft_summary)

    if per_ds_summaries:
        for ds_name, s in per_ds_summaries.items():
            print(f"\n  ── {ds_name} ──")
            _print_summary(s)

    # ── 7. 保存汇总 JSON ─────────────────────────────────────────
    summary_data = {
        "finetuned":     ft_summary,
        "dataset_names": dataset_names,
    }
    if per_ds_summaries:
        summary_data["per_dataset"] = per_ds_summaries

    summary_file = output_dir / f"{ckpt_name}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"  详细结果已保存至: {ft_output}")
    print(f"  汇总 JSON:         {summary_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

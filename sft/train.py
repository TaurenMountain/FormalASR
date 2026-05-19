"""
sft/train.py
SFT 训练入口。数据集直接从 HuggingFace Hub 下载，无需本地 JSONL 文件。

运行示例：
    CUDA_VISIBLE_DEVICES=0,1 python -m sft.train \\
        --model_path TaurenMountain/FormalASR-1.7B \\
        --dataset_name wenetspeech \\
        --output_dir checkpoints/formalASR-1.7b-sft \\
        --epochs 1 --batch_size 4 --grad_acc 8
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset as HFDataset
from transformers import GenerationConfig, TrainingArguments

from common.model_utils import patch_outer_forward
from .data import (
    DataCollatorForQwen3ASRFinetuning,
    build_records_from_hf_dataset,
    make_preprocess_fn_prefix_only,
)
from .model import (
    CastFloatInputsTrainer,
    MakeEveryCheckpointInferableCallback,
    find_latest_checkpoint,
)

_THIS_DIR = Path(__file__).resolve().parent   # sft/


# ──────────────────────────────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("FormalASR SFT 全量微调")

    # 模型
    p.add_argument("--model_path", type=str,
                   default="TaurenMountain/FormalASR-1.7B",
                   help="HuggingFace 模型 ID 或本地模型目录")

    # 数据集
    p.add_argument("--dataset_name", type=str,
                   default="TaurenMountain/WenetSpeech-Formal",
                   help="HuggingFace 数据集 repo ID，例如："
                        "TaurenMountain/WenetSpeech-Formal 或 TaurenMountain/Speechio-Formal")
    p.add_argument("--hf_cache_dir", type=str, default=None,
                   help="HuggingFace 数据集缓存目录（None 则使用默认 ~/.cache/huggingface）")

    # 输出
    p.add_argument("--output_dir", type=str,
                   default=str(_THIS_DIR.parent / "checkpoints" / "formalASR-sft"),
                   help="checkpoint 输出目录")

    # 数据限制
    p.add_argument("--max_samples",      type=int, default=-1,
                   help="最多加载多少条训练样本（-1=全部）")
    p.add_argument("--max_eval_samples", type=int, default=2000,
                   help="最多加载多少条验证样本（默认 2000）")

    # 训练超参
    p.add_argument("--batch_size",        type=int,   default=4)
    p.add_argument("--grad_acc",          type=int,   default=8)
    p.add_argument("--lr",                type=float, default=2e-5)
    p.add_argument("--epochs",            type=float, default=1)
    p.add_argument("--log_steps",         type=int,   default=10)
    p.add_argument("--lr_scheduler_type", type=str,   default="cosine")
    p.add_argument("--warmup_ratio",      type=float, default=0.05)

    # DataLoader
    p.add_argument("--num_workers",        type=int, default=0)
    p.add_argument("--pin_memory",         type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=0)
    p.add_argument("--prefetch_factor",    type=int, default=2)

    # 保存策略
    p.add_argument("--save_strategy",    type=str, default="steps")
    p.add_argument("--save_steps",       type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)

    # 断点续训
    p.add_argument("--resume_from", type=str, default="",
                   help="指定 checkpoint 目录续训")
    p.add_argument("--resume",      type=int, default=0,
                   help="1=自动查找最新 checkpoint 续训")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    ds_name: str = args.dataset_name.strip()
    if not ds_name:
        raise ValueError("--dataset_name 不能为空")

    print("=" * 60)
    print("  FormalASR SFT 全量微调")
    print(f"  模型路径:       {args.model_path}")
    print(f"  数据集:         {ds_name}")
    print(f"  输出目录:       {args.output_dir}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  批大小:         {args.batch_size}  梯度累积: {args.grad_acc}")
    print(f"  等效全局 batch: {args.batch_size * args.grad_acc}")
    print(f"  学习率:         {args.lr}")
    print("=" * 60)

    # 1. 加载模型
    from qwen_asr import Qwen3ASRModel
    print(f"\n[模型] 加载 {args.model_path}，dtype={'bfloat16' if use_bf16 else 'float16'} ...")
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model     = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[模型] gradient_checkpointing 已开启 ✓")
    elif hasattr(model, "thinker") and hasattr(model.thinker, "gradient_checkpointing_enable"):
        model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[模型] thinker.gradient_checkpointing 已开启 ✓")

    print("[模型] 加载完成 ✓")

    # 2. 从 HuggingFace Hub 加载数据集并构建记录
    print("\n[数据] 从 HuggingFace Hub 加载数据集...")
    from common.audio import load_hf_dataset

    hf_datasets: Dict[str, any] = {}

    # 训练集
    print(f"\n  [{ds_name}] 加载 train split...")
    train_hf_ds = load_hf_dataset(ds_name, "train", cache_dir=args.hf_cache_dir)
    hf_datasets[f"{ds_name}:train"] = train_hf_ds

    train_records: List[dict] = build_records_from_hf_dataset(
        train_hf_ds, dataset_name=ds_name, split="train",
        max_samples=args.max_samples,
    )
    print(f"    → {len(train_records)} 条训练样本")

    if len(train_records) == 0:
        raise RuntimeError("训练数据为空，请检查数据集")

    # 验证集（可选）
    eval_records: List[dict] = []
    try:
        print(f"  [{ds_name}] 加载 validation split...")
        val_hf_ds = load_hf_dataset(ds_name, "validation", cache_dir=args.hf_cache_dir)
        hf_datasets[f"{ds_name}:validation"] = val_hf_ds

        eval_records = build_records_from_hf_dataset(
            val_hf_ds, dataset_name=ds_name, split="validation",
            max_samples=args.max_eval_samples,
        )
        print(f"    → {len(eval_records)} 条验证样本")
    except Exception as e:
        print(f"  [{ds_name}] 未找到 validation split，跳过: {e}")

    # 3. 构建 prefix_text
    print("\n[数据] 构建 chat template prefix...")
    preprocess_fn = make_preprocess_fn_prefix_only(processor)

    raw_train = HFDataset.from_list(train_records)
    train_ds  = raw_train.map(preprocess_fn, num_proc=1, desc="preprocess train")

    eval_ds = None
    if eval_records:
        raw_eval = HFDataset.from_list(eval_records)
        eval_ds  = raw_eval.map(preprocess_fn, num_proc=1, desc="preprocess eval")

    print("  数据预处理完成 ✓")

    # 4. DataCollator
    collator = DataCollatorForQwen3ASRFinetuning(
        processor=processor,
        hf_datasets=hf_datasets,
    )

    # 5. TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=(args.pin_memory == 1),
        dataloader_persistent_workers=(args.persistent_workers == 1 and args.num_workers > 0),
        dataloader_prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.save_steps if eval_ds is not None else None,
        do_eval=(eval_ds is not None),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
    )

    # 6. 创建 Trainer
    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[
            MakeEveryCheckpointInferableCallback(base_model_path=args.model_path)
        ],
    )

    # 7. 断点续训
    resume_from = (args.resume_from or "").strip()
    if not resume_from and args.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    # 8. 开始训练
    print(f"\n[训练] 开始！{'从 ' + resume_from + ' 恢复' if resume_from else '从头开始'}")
    if resume_from:
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    print("\n" + "=" * 60)
    print("  训练完成！")
    print(f"  checkpoint 保存在: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

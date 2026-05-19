"""
grpo/train.py
GRPO 强化学习训练入口。数据集直接从 HuggingFace Hub 下载，无需本地 JSONL 文件。

运行示例：
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch -m grpo.train \\
        --model_path TaurenMountain/FormalASR-1.7B \\
        --dataset_name wenetspeech \\
        --output_dir checkpoints/formalASR-1.7b-grpo \\
        --epochs 1 --batch_size 1 --num_generations 4 --kl_coef 0.01
"""
from __future__ import annotations

import argparse
import copy
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset as HFDataset
from tqdm import tqdm
from transformers import GenerationConfig

from common.audio import decode_audio_from_hf, load_hf_dataset
from common.model_utils import patch_outer_forward
from .reward import QwenJudgeScorer, build_reward_fn
from .trainer import AudioBatchBuilder, MinimalGRPOTrainer

_THIS_DIR = Path(__file__).resolve().parent   # grpo/


# ──────────────────────────────────────────────────────────────────────
# 从 HF Dataset 构建训练记录
# ���─────────────────────────────────────────────────────────────────────

def build_grpo_records(
    hf_dataset,
    dataset_name: str,
    split: str = "train",
    max_samples: int = -1,
    language: str = "Chinese",
) -> List[Dict[str, Any]]:
    """
    从 HF Dataset 对象直接构建 GRPO 训练记录。
    每条记录包含：idx、label、language、split、dataset_name
    音频在 AudioBatchBuilder 中按 idx + dataset_name:split 键从 hf_datasets 动态读取。
    """
    records = []
    n = len(hf_dataset)
    if max_samples > 0:
        n = min(n, max_samples)

    for i in range(n):
        row = hf_dataset[i]
        tgt = row.get("target_text", "")
        if not tgt:
            continue
        records.append({
            "idx":          i,
            "label":        tgt,
            "language":     row.get("language", language),
            "split":        row.get("split", split),
            "dataset_name": dataset_name,
        })

    return records


# ──────────────────────────────────────────────────────────────────────
# Checkpoint 工具
# ──────────────────────────────────────────────────────────────────────

def save_checkpoint(
    output_dir: str,
    tag: str,
    model,
    processor,
    optimizer,
    scheduler,
    step: int,
    global_batch: int = 0,
):
    ckpt = os.path.join(output_dir, f"checkpoint-{tag}")
    os.makedirs(ckpt, exist_ok=True)
    model.save_pretrained(ckpt, safe_serialization=True)
    processor.save_pretrained(ckpt)
    torch.save(
        {"step": step, "global_batch": global_batch},
        os.path.join(ckpt, "trainer_state.pt"),
    )
    torch.save(optimizer.state_dict(), os.path.join(ckpt, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(ckpt, "scheduler.pt"))
    with open(os.path.join(output_dir, "latest_checkpoint.txt"), "w", encoding="utf-8") as f:
        f.write(ckpt)


def rotate_checkpoints(output_dir: str, save_total_limit: int):
    """仅保留最近 N 个 checkpoint-<step>，不处理 checkpoint-final。"""
    if not save_total_limit or save_total_limit <= 0:
        return
    import re
    import shutil
    items = []
    for name in os.listdir(output_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        if not m:
            continue
        ckpt_path = os.path.join(output_dir, name)
        if os.path.isdir(ckpt_path):
            items.append((int(m.group(1)), ckpt_path))
    if len(items) <= save_total_limit:
        return
    items.sort(key=lambda x: x[0])
    for _, ckpt_path in items[: len(items) - save_total_limit]:
        import shutil
        shutil.rmtree(ckpt_path, ignore_errors=True)


def resolve_resume_path(args) -> str:
    ckpt = (args.resume_from_checkpoint or "").strip()
    if not ckpt:
        return ""
    if ckpt.lower() == "latest":
        p = os.path.join(args.output_dir, "latest_checkpoint.txt")
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read().strip()
        return ""
    return ckpt if os.path.isdir(ckpt) else ""


# ──────────────────────────────────────────────────���───────────────────
# 参数解析
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("FormalASR Minimal GRPO")

    p.add_argument("--model_path",     type=str,
                   default="TaurenMountain/FormalASR-1.7B",
                   help="HuggingFace 模型 ID 或本地 checkpoint 目录")
    p.add_argument("--sft_checkpoint", type=str, default="",
                   help="SFT checkpoint 目录（优先于 model_path）")
    p.add_argument("--dataset_name",   type=str, default="wenetspeech",
                   help="数据集名称（别名或完整 HF repo ID）")
    p.add_argument("--hf_cache_dir",   type=str, default=None,
                   help="HuggingFace 数据集缓存目录")
    p.add_argument("--output_dir",     type=str,
                   default=str(_THIS_DIR.parent / "checkpoints" / "formalASR-grpo"))

    p.add_argument("--max_samples",            type=int,   default=-1)
    p.add_argument("--batch_size",             type=int,   default=1)
    p.add_argument("--num_generations",        type=int,   default=4)
    p.add_argument("--max_completion_length",  type=int,   default=80)
    p.add_argument("--grad_acc",               type=int,   default=4)
    p.add_argument("--lr",                     type=float, default=5e-7)
    p.add_argument("--epochs",                 type=float, default=1.0)
    p.add_argument("--total_steps",            type=int,   default=0)
    p.add_argument("--kl_coef",                type=float, default=0.01,
                   help="KL 散度惩罚系数（0=不加 ref_model 约束，节省显存）")
    p.add_argument("--temperature",            type=float, default=0.9)
    p.add_argument("--top_p",                  type=float, default=0.95)
    p.add_argument("--warmup_ratio",           type=float, default=0.05)

    p.add_argument("--reward_cer_weight",         type=float, default=1.0)
    p.add_argument("--reward_judge_weight",       type=float, default=0.0)
    p.add_argument("--reward_judge_url",          type=str,
                   default="http://localhost:9500/v1/chat/completions")
    p.add_argument("--reward_judge_model",        type=str,   default="qwen35-27b")
    p.add_argument("--reward_judge_timeout",      type=float, default=20.0)
    p.add_argument("--reward_judge_fail_value",   type=float, default=0.0)

    p.add_argument("--resume_from_checkpoint", type=str, default="")
    p.add_argument("--log_steps",              type=int, default=10)
    p.add_argument("--save_steps",             type=int, default=100)
    p.add_argument("--seed",                   type=int, default=42)
    p.add_argument("--save_total_limit",       type=int, default=3)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        accelerator = Accelerator(
            mixed_precision="bf16" if use_bf16 else "fp16",
            gradient_accumulation_steps=args.grad_acc,
        )
        set_seed(args.seed)
    except Exception:
        accelerator = None

    is_main   = (accelerator is None) or accelerator.is_main_process
    num_procs = accelerator.num_processes if accelerator else 1

    # 加载模型
    from qwen_asr import Qwen3ASRModel
    model_path  = (args.sft_checkpoint or args.model_path).strip()
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map=None,
        attn_implementation="sdpa",
    )
    model     = asr_wrapper.model
    processor = asr_wrapper.processor
    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    for param in model.parameters():
        param.requires_grad = True

    ref_model = None
    if args.kl_coef > 0:
        ref_model = copy.deepcopy(model)
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()

    # 从 HF Hub 加载数据集
    if is_main:
        print(
            f"\n{'='*60}\n"
            f"[启动] 模型路径   : {model_path}\n"
            f"[启动] 输出目录   : {args.output_dir}\n"
            f"[启动] 数据集     : {args.dataset_name}\n"
            f"[启动] lr={args.lr}  kl_coef={args.kl_coef}  "
            f"batch={args.batch_size}  grad_acc={args.grad_acc}  "
            f"num_gen={args.num_generations}\n"
            f"{'='*60}\n"
        )

    print(f"[数据] 从 HuggingFace Hub 加载 {args.dataset_name} train split...")
    train_hf_ds = load_hf_dataset(
        args.dataset_name, "train", cache_dir=args.hf_cache_dir
    )
    records = build_grpo_records(
        train_hf_ds,
        dataset_name=args.dataset_name,
        split="train",
        max_samples=args.max_samples,
    )
    if not records:
        raise RuntimeError("训练数据为空，请检查数据集")

    print(f"[数据] 共 {len(records)} 条训练样本")

    # hf_datasets 以 "dataset_name:split" 为 key
    hf_datasets: Dict[str, Any] = {f"{args.dataset_name}:train": train_hf_ds}

    dataset = HFDataset.from_list(records)

    steps_per_epoch = max(1, len(dataset) // (num_procs * args.batch_size * args.grad_acc))
    if args.total_steps <= 0:
        args.total_steps = max(1, int(steps_per_epoch * args.epochs))

    # 设备分配
    if accelerator:
        model = accelerator.prepare(model)
        if ref_model is not None:
            ref_model = ref_model.to(accelerator.device)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = model.to(device)
        if ref_model is not None:
            ref_model = ref_model.to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    builder = AudioBatchBuilder(processor, hf_datasets)

    # 奖励函数
    judge_scorer = None
    if args.reward_judge_weight > 0:
        judge_scorer = QwenJudgeScorer(
            url=args.reward_judge_url,
            model=args.reward_judge_model,
            timeout=args.reward_judge_timeout,
            fail_value=args.reward_judge_fail_value,
        )
        if is_main:
            print(
                f"[Reward] 启用 qwen-judge: url={args.reward_judge_url}, "
                f"model={args.reward_judge_model}, "
                f"cer_w={args.reward_cer_weight}, judge_w={args.reward_judge_weight}"
            )

    reward_fn = build_reward_fn(
        cer_weight=args.reward_cer_weight,
        judge_weight=args.reward_judge_weight,
        judge_scorer=judge_scorer,
    )
    trainer = MinimalGRPOTrainer(
        model, ref_model, processor, builder, args, accelerator, reward_fn=reward_fn
    )

    # 断点恢复
    resume_path  = resolve_resume_path(args)
    step         = 0
    global_batch = 0
    if resume_path:
        opt_path   = os.path.join(resume_path, "optimizer.pt")
        sch_path   = os.path.join(resume_path, "scheduler.pt")
        state_path = os.path.join(resume_path, "trainer_state.pt")
        if os.path.isfile(opt_path):
            trainer.optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
        if os.path.isfile(sch_path):
            trainer.scheduler.load_state_dict(torch.load(sch_path, map_location="cpu"))
        if os.path.isfile(state_path):
            state        = torch.load(state_path, map_location="cpu")
            step         = int(state.get("step", 0))
            global_batch = int(state.get("global_batch", step * args.grad_acc))
        if is_main:
            print(f"[Resume] 从 {resume_path} 恢复，step={step}，global_batch={global_batch}")

    # 训练循环
    all_records   = [dict(dataset[i]) for i in range(len(dataset))]
    pbar          = tqdm(total=args.total_steps, desc="RL Training", disable=not is_main)
    pbar.update(step)

    trainer.optimizer.zero_grad()
    accum         = global_batch
    batch_counter = 0
    epoch         = 0

    while step < args.total_steps:
        epoch += 1
        random.seed(args.seed + epoch)
        random.shuffle(all_records)

        rank  = accelerator.process_index  if accelerator else 0
        world = accelerator.num_processes  if accelerator else 1
        shard = all_records[rank::world]

        for st in range(0, len(shard), args.batch_size):
            if step >= args.total_steps:
                break
            batch = shard[st: st + args.batch_size]
            if not batch:
                continue

            if batch_counter < global_batch:
                batch_counter += 1
                accum += 1
                continue

            will_update   = ((accum + 1) % args.grad_acc == 0)
            next_step     = step + 1 if will_update else step
            do_log_sample = will_update and is_main and (next_step % args.log_steps == 0 or next_step == 1)

            logs          = trainer.train_step(batch, log_sample=do_log_sample, step=next_step)
            accum         += 1
            batch_counter += 1

            if accum % args.grad_acc == 0:
                if accelerator:
                    accelerator.clip_grad_norm_(
                        [p for p in accelerator.unwrap_model(trainer.model).parameters() if p.requires_grad],
                        1.0,
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in trainer.model.parameters() if p.requires_grad], 1.0
                    )

                trainer.optimizer.step()
                trainer.scheduler.step()
                trainer.optimizer.zero_grad()
                step += 1

                if is_main and (step % args.log_steps == 0 or step == 1):
                    lr = trainer.scheduler.get_last_lr()[0]
                    pbar.write(
                        f"[step {step}/{args.total_steps}] loss={logs['loss']:.4f} "
                        f"pg={logs['pg_loss']:.4f} kl={logs['kl_loss']:.4f} "
                        f"reward={logs['reward']:.4f} cer={logs['cer']:.4f} "
                        f"judge={logs['judge']:.4f} lr={lr:.2e}"
                    )

                if is_main and step % args.save_steps == 0:
                    raw = accelerator.unwrap_model(trainer.model) if accelerator else trainer.model
                    save_checkpoint(
                        args.output_dir, str(step), raw, processor,
                        trainer.optimizer, trainer.scheduler,
                        step, global_batch=accum,
                    )
                    rotate_checkpoints(args.output_dir, args.save_total_limit)

                if is_main:
                    pbar.update(1)

    pbar.close()

    if is_main:
        raw = accelerator.unwrap_model(trainer.model) if accelerator else trainer.model
        save_checkpoint(
            args.output_dir, "final", raw, processor,
            trainer.optimizer, trainer.scheduler,
            step, global_batch=accum,
        )
        print(f"[完成] 训练结束，总步数: {step}，输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

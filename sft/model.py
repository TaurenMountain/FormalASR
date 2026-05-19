"""
sft/model.py
SFT 模型工具：Trainer 子类、Checkpoint 管理与保存回调。
"""
from __future__ import annotations

import os
import re
import shutil
from typing import Optional

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments


# ──────────────────────────────────────────────────────────────────────
# Checkpoint 管理
# ──────────────────────────────────────────────────────────────────────

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """在 output_dir 中查找步数最大的 checkpoint-N 目录。"""
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step, best_path = None, None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step, best_path = step, path
    return best_path


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    """
    将基础模型的配置文件复制到 checkpoint 目录，使其可直接用于推理。
    """
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


# ──────────────────────────────────────────────────────────────────────
# Trainer 子类：自动将浮点输入 cast 到模型 dtype
# ──────────────────────────────────────────────────────────────────────

class CastFloatInputsTrainer(Trainer):
    """自动将 input_features 等浮点 Tensor cast 到模型 dtype，避免 bf16/fp16 不匹配。"""

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


# ──────────────────────────────────────────────────────────────────────
# Callback：每次 save 后复制基础模型配置文件到 checkpoint
# ──────────────────────────────────────────────────────────────────────

class MakeEveryCheckpointInferableCallback(TrainerCallback):
    """每次 checkpoint 保存后，将基础模型的配置文件复制到 checkpoint 目录，使其可直接推理。"""

    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)
        copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
        return control

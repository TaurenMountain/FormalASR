"""
sft/data.py
SFT 数据处理：从 HuggingFace Dataset 构建训练记录、构建 chat template prefix，
以及用于训练的 DataCollator。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from common.audio import TARGET_SR, decode_audio_from_hf


# ──────────────────────────────────────────────────────────────────────
# 从 HF Dataset 构建训练记录
# ──────────────────────────────────────────────────────────────────────

def build_records_from_hf_dataset(
    hf_dataset,
    dataset_name: str,
    split: str = "train",
    max_samples: int = -1,
    language: str = "Chinese",
) -> List[Dict]:
    """
    从 HF Dataset 对象直接构建训练记录列表。
    每条记录包含：idx、target、language、split、dataset_name
    音频在 DataCollator 中按 idx 从 hf_dataset 动态读取。
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
            "idx":          i,
            "target":       target,
            "language":     row.get("language", language),
            "split":        row.get("split", split),
            "dataset_name": dataset_name,
        })

    return records


# ──────────────────────────────────────────────────────────────────────
# Chat Template Prefix 构建
# ──────────────────────────────────────────────────────────────────────

def build_prefix_messages(prompt: str, audio_array):
    """构建用于 apply_chat_template 的消息列表（含音频占位符）。"""
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    """
    返回一个预处理函数，将记录转换为包含 prefix_text 的字典。
    prefix_text = apply_chat_template 结果 + "language {lang}<asr_text>"，
    与 qwen_asr.transcribe(language=...) 行为一致。
    """
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        language = ex.get("language", "Chinese")
        prefix_text = prefix_text + f"language {language}<asr_text>"
        return {**ex, "prefix_text": prefix_text}
    return _preprocess


# ──────────────────────────────────────────────────────────────────────
# DataCollator
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DataCollatorForQwen3ASRFinetuning:
    """
    从 HF Dataset 按 idx 动态读取音频，并组装训练 batch。

    hf_datasets 格式：{"dataset_name:split": HF Dataset, ...}
    """
    processor: Any
    hf_datasets: Dict[str, Any]

    def _lookup_hf_dataset(self, f: Dict[str, Any]):
        """根据记录中的 dataset_name / split 字段查找对应的 HF Dataset。"""
        split    = f.get("split", "train")
        ds_name  = f.get("dataset_name", "")

        composite_key = f"{ds_name}:{split}" if ds_name else split
        if composite_key in self.hf_datasets:
            return self.hf_datasets[composite_key]

        # 降级：尝试纯 split 键
        if split in self.hf_datasets:
            return self.hf_datasets[split]

        available = list(self.hf_datasets.keys())
        raise KeyError(
            f"hf_datasets 中找不到对应的数据集。\n"
            f"  dataset_name={ds_name!r}, split={split!r}\n"
            f"  尝试的键: {composite_key!r}\n"
            f"  可用键: {available}"
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prefix_texts = [f["prefix_text"] for f in features]
        targets      = [f["target"] for f in features]

        audios = []
        for f in features:
            try:
                ds  = self._lookup_hf_dataset(f)
                row = ds[int(f["idx"])]
                audio_array = decode_audio_from_hf(row["audio"])
            except Exception as e:
                warnings.warn(f"[Collator] 读取音频失败 idx={f.get('idx')}: {e}，使用静音占位")
                audio_array = np.zeros(TARGET_SR, dtype=np.float32)
            audios.append(audio_array)

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        # 仅对 target+eos 做纯文本 tokenize，推算 prefix 长度
        target_eos_texts = [tgt + eos for tgt in targets]
        target_enc = self.processor.tokenizer(
            target_eos_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        target_lens     = target_enc["attention_mask"].sum(dim=1)
        full_valid_lens = full_inputs["attention_mask"].sum(dim=1)
        prefix_lens     = (full_valid_lens - target_lens).clamp(min=0).tolist()

        # 只计算 target 部分的 loss
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :int(pl)] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs

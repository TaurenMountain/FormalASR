"""
common/audio.py
音频解码与 HuggingFace Dataset 加载工具。
被 sft、grpo、eval 三个模块共同使用。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

TARGET_SR = 16000

# ──────────────────────────────────────────────────────────────────────
# 数据集名称 → HuggingFace 仓库 ID 映射
# ──────────────────────────────────────────────────────────────────────

DATASET_HF_IDS: Dict[str, str] = {
    "wenetspeech": "TaurenMountain/WenetSpeech-Formal",
    "speechio":    "TaurenMountain/Speechio-Formal",
}


def resolve_hf_dataset_id(dataset_name: str) -> str:
    """
    将数据集名称解析为 HuggingFace 仓库 ID。
    - 若 dataset_name 包含 '/'，直接视为完整 HF repo ID。
    - 否则在 DATASET_HF_IDS 中查找映射；找不到则抛出 ValueError。
    """
    if "/" in dataset_name:
        return dataset_name
    if dataset_name in DATASET_HF_IDS:
        return DATASET_HF_IDS[dataset_name]
    raise ValueError(
        f"未知数据集名称: {dataset_name!r}。\n"
        f"  已知映射: {list(DATASET_HF_IDS.keys())}\n"
        f"  也可直接传入完整 HF repo ID，如 'TaurenMountain/WenetSpeech-Formal'。"
    )


# ──────────────────────────────────────────────────────────────────────
# 音频解码
# ──────────────────────────────────────────────────────────────────────

def decode_audio_from_hf(audio_data) -> np.ndarray:
    """
    将 HF Dataset 中的 audio 字段解码为 float32 numpy 数组（16kHz 单通道）。
    支持：
      - AudioDecoder（torchcodec 懒加载，datasets >= 3.x）
      - dict {"array": np.ndarray, "sampling_rate": int}（旧版 datasets）
    """
    if hasattr(audio_data, "get_all_samples"):
        samples = audio_data.get_all_samples()
        tensor = samples.data          # shape: [C, T]
        sr = int(samples.sample_rate)
        tensor = tensor.mean(dim=0) if tensor.shape[0] > 1 else tensor[0]
        array = tensor.cpu().numpy().astype(np.float32)
    elif isinstance(audio_data, dict):
        array = np.array(audio_data["array"], dtype=np.float32)
        sr = int(audio_data["sampling_rate"])
        if array.ndim == 2:
            array = array.mean(axis=1)
    else:
        array = np.array(audio_data, dtype=np.float32)
        sr = TARGET_SR

    if sr != TARGET_SR:
        try:
            import librosa
            array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
        except ImportError:
            ratio = TARGET_SR / sr
            new_len = max(1, int(len(array) * ratio))
            indices = np.linspace(0, len(array) - 1, new_len)
            array = np.interp(indices, np.arange(len(array)), array).astype(np.float32)

    return array.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# HuggingFace Dataset 加载
# ──────────────────────────────────────────────────────────────────────

def load_hf_dataset(
    dataset_name: str,
    split: str,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """
    从 HuggingFace Hub 加载指定数据集的某个 split。

    参数：
      dataset_name  本地别名（如 "wenetspeech"）或完整 HF repo ID
                    （如 "TaurenMountain/WenetSpeech-Formal"）
      split         split 名称，如 "train" / "validation" / "test"
      cache_dir     HF 缓存目录（None 则使用 HF 默认缓存 ~/.cache/huggingface）
      trust_remote_code  是否信任远程代码（通常不需要）
    """
    from datasets import load_dataset

    repo_id = resolve_hf_dataset_id(dataset_name)
    print(f"[数据] 从 HuggingFace 加载 {repo_id}  split='{split}' ...")
    ds = load_dataset(repo_id, split=split, cache_dir=cache_dir,
                      trust_remote_code=trust_remote_code)
    print(f"  → {len(ds)} 条样本，列: {ds.column_names}")
    return ds


def load_hf_splits(
    dataset_name: str,
    splits: List[str],
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    加载指定 split 列表，返回 {split_name: dataset} 字典。
    多个逻辑 split 映射到同一 HF split（如 test-meeting / test-net → test）时共享同一对象。
    """
    split_cache: Dict[str, Any] = {}
    result: Dict[str, Any] = {}

    for split in splits:
        hf_split = _normalize_split(split)
        if hf_split not in split_cache:
            split_cache[hf_split] = load_hf_dataset(
                dataset_name, hf_split, cache_dir=cache_dir
            )
        result[split] = split_cache[hf_split]
    return result


def _normalize_split(split: str) -> str:
    """将 JSONL 的 split 字段映射到 HF split 名称。"""
    s = (split or "train").lower()
    if s.startswith("test"):
        return "test"
    if s.startswith("val") or s.startswith("dev"):
        return "validation"
    return "train"

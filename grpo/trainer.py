"""
grpo/trainer.py
GRPO 训练核心：音频批处理构建器（AudioBatchBuilder）和最小化 GRPO 训练器（MinimalGRPOTrainer）。
"""
from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from common.audio import TARGET_SR, decode_audio_from_hf
from .reward import cer_reward


# ─────────────────────────────────────────────────────────────────────
# 音频批处理构建器
# ─────────────────────────────────────────────────────────────────────

class AudioBatchBuilder:
    """
    负责从 HF Dataset 读取音频，构建推理 prompt 输入和训练完整输入。
    """

    def __init__(self, processor, hf_datasets: Optional[Dict[str, Any]] = None):
        self.processor    = processor
        self.hf_datasets  = hf_datasets or {}

    @staticmethod
    def _decode_audio_from_path(audio_path: str) -> np.ndarray:
        from pathlib import Path

        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        try:
            import soundfile as sf
            array, sr = sf.read(str(p), dtype="float32", always_2d=False)
            if isinstance(array, np.ndarray) and array.ndim == 2:
                array = array.mean(axis=1)
        except Exception:
            import wave
            with wave.open(str(p), "rb") as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)

            if sampwidth == 2:
                array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                array = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise RuntimeError(f"不支持的 WAV 采样宽度: {sampwidth}")

            if n_channels > 1:
                array = array.reshape(-1, n_channels).mean(axis=1)

        if sr != TARGET_SR:
            try:
                import librosa
                array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
            except Exception:
                ratio = TARGET_SR / float(sr)
                new_len = max(1, int(len(array) * ratio))
                indices = np.linspace(0, len(array) - 1, new_len)
                array = np.interp(indices, np.arange(len(array)), array).astype(np.float32)

        return np.asarray(array, dtype=np.float32)

    def _read_audio(self, feature: Dict[str, Any]) -> np.ndarray:
        if not self.hf_datasets:
            raise RuntimeError("hf_datasets 为空，无法读取音频")

        idx     = feature["idx"]
        split   = feature.get("split", "train")
        ds_name = feature.get("dataset_name", "")

        # 优先用 "dataset_name:split" 复合键，降级到纯 split 键
        composite = f"{ds_name}:{split}" if ds_name else split
        ds = self.hf_datasets.get(composite) or self.hf_datasets.get(split)
        if ds is None:
            available = list(self.hf_datasets.keys())
            raise KeyError(
                f"hf_datasets 中找不到键 {composite!r}，可用键: {available}"
            )
        row = ds[idx]
        return decode_audio_from_hf(row["audio"])

    def build_prompt_inputs(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建推理阶段的模型输入（不含 completion）。"""
        audios = [self._read_audio(x) for x in features]
        messages = [
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": [{"type": "audio", "audio": audio}]},
            ]
            for audio in audios
        ]
        prompt_texts = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        # 追加 language 强制前缀，与 qwen_asr.transcribe(language=...) 行为一致
        prefixes = [f"language {x.get('language', 'Chinese')}<asr_text>" for x in features]
        prompt_texts = [p + s for p, s in zip(prompt_texts, prefixes)]

        inputs = self.processor(
            text=prompt_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        inputs["_prompt_texts"] = prompt_texts
        return inputs

    def build_train_inputs(
        self,
        prompt_inputs: Dict[str, Any],
        completions: List[str],
        group_size: int,
    ) -> Dict[str, Any]:
        """构建训练阶段的完整输入（prompt + completion），并生成 labels mask。"""
        eos          = self.processor.tokenizer.eos_token or ""
        prompt_texts = prompt_inputs["_prompt_texts"]
        rep_prompts  = [p for p in prompt_texts for _ in range(group_size)]
        full_texts   = [p + c + eos for p, c in zip(rep_prompts, completions)]

        tok = self.processor.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        labels = tok["input_ids"].clone()

        target_tok = self.processor.tokenizer(
            [c + eos for c in completions],
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        target_lens = target_tok["attention_mask"].sum(dim=1)
        full_lens   = tok["attention_mask"].sum(dim=1)
        prompt_lens = (full_lens - target_lens).clamp(min=0)

        seq_len = labels.shape[1]
        pos     = torch.arange(seq_len).unsqueeze(0)
        labels[pos < prompt_lens.unsqueeze(1)] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        out = dict(tok)
        out["labels"]      = labels
        out["prompt_lens"] = prompt_lens
        for k in ("input_features", "feature_attention_mask"):
            if k in prompt_inputs and torch.is_tensor(prompt_inputs[k]):
                out[k] = prompt_inputs[k].repeat_interleave(group_size, dim=0)
        return out


# ──────────────────────────────────────────────────────────────────────
# GRPO 训练器
# ──────────────────────────────────────────────────────────────────────

class MinimalGRPOTrainer:
    """
    最小化 GRPO 训练器：采样 → 奖励 → 组内优势标准化 → 策略梯度更新（+ KL 惩罚）。
    支持单卡和 accelerate 多卡。
    """

    def __init__(
        self,
        model,
        ref_model,
        processor,
        builder: AudioBatchBuilder,
        args,
        accelerator=None,
        reward_fn: Optional[Callable] = None,
    ):
        self.model       = model
        self.ref_model   = ref_model
        self.processor   = processor
        self.builder     = builder
        self.args        = args
        self.accelerator = accelerator
        self.device      = accelerator.device if accelerator else next(model.parameters()).device
        self.reward_fn   = reward_fn if reward_fn is not None else (
            lambda h, r: [cer_reward(hi, ri) for hi, ri in zip(h, r)]
        )

        raw_model = accelerator.unwrap_model(model) if accelerator else model
        params    = [p for p in raw_model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=max(1, int(args.total_steps * args.warmup_ratio)),
            num_training_steps=max(1, args.total_steps),
        )
        if accelerator:
            self.optimizer, self.scheduler = accelerator.prepare(self.optimizer, self.scheduler)

    @torch.inference_mode()
    def generate_completions(
        self, batch_features: List[Dict[str, Any]]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """对一批样本生成 G 条采样 completion。"""
        g             = self.args.num_generations
        prompt_inputs = self.builder.build_prompt_inputs(batch_features)
        prompt_inputs = {
            k: (v.to(self.device) if torch.is_tensor(v) else v)
            for k, v in prompt_inputs.items()
        }

        expanded = {
            k: v.repeat_interleave(g, dim=0) if (torch.is_tensor(v) and k != "_prompt_texts") else v
            for k, v in prompt_inputs.items()
            if k != "_prompt_texts"
        }

        raw_model   = self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
        model_dtype = getattr(raw_model, "dtype", None)
        if model_dtype is not None and "input_features" in expanded and torch.is_tensor(expanded["input_features"]):
            expanded["input_features"] = expanded["input_features"].to(dtype=model_dtype)

        thinker = raw_model.thinker if hasattr(raw_model, "thinker") else raw_model
        eos_id  = self.processor.tokenizer.eos_token_id

        out_ids = thinker.generate(
            **expanded,
            max_new_tokens=self.args.max_completion_length,
            do_sample=self.args.temperature > 0,
            temperature=self.args.temperature if self.args.temperature > 0 else None,
            top_p=self.args.top_p if self.args.temperature > 0 else None,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=eos_id,
        )

        prompt_len = expanded["input_ids"].shape[1]
        gen_ids    = out_ids[:, prompt_len:]
        texts      = self.processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return [t.strip() for t in texts], prompt_inputs

    def _log_samples(
        self,
        completions: List[str],
        labels: List[str],
        rewards_list: List[float],
        group_size: int,
        step: int,
        cer_scores: Optional[List[float]] = None,
        judge_scores: Optional[List[float]] = None,
    ):
        """打印第一个 prompt 的 G 条采样结果（ref / hyp / reward / cer / judge）。"""
        g   = group_size
        ref = labels[0]
        sep = "=" * 64
        print(f"\n{sep}", file=sys.stderr, flush=True)
        print(f"[Sample @ step {step}] ref ({len(ref)}c): {ref[:150]!r}", file=sys.stderr, flush=True)
        for i in range(min(g, len(completions))):
            hyp = completions[i]
            r   = rewards_list[i]
            c   = cer_scores[i]   if cer_scores   is not None and i < len(cer_scores)   else 0.0
            j   = judge_scores[i] if judge_scores is not None and i < len(judge_scores) else 0.0
            print(
                f"  hyp[{i}] reward={r:+.3f} cer={c:+.3f} judge={j:+.3f}  len={len(hyp)}c  {hyp[:150]!r}",
                file=sys.stderr, flush=True,
            )
        print(sep, file=sys.stderr, flush=True)

    def train_step(
        self,
        batch_features: List[Dict[str, Any]],
        log_sample: bool = False,
        step: int = 0,
    ) -> Dict[str, float]:
        """单步训练：生成 → 计算奖励 → 梯度（不执行 optimizer.step）。"""
        g = self.args.num_generations
        b = len(batch_features)

        completions, prompt_inputs = self.generate_completions(batch_features)
        labels = []
        for x in batch_features:
            labels.extend([x["label"]] * g)

        try:
            rewards_list, cer_scores, judge_scores = self.reward_fn(completions, labels, return_parts=True)
        except TypeError:
            rewards_list  = self.reward_fn(completions, labels)
            cer_scores    = [0.0] * len(rewards_list)
            judge_scores  = [0.0] * len(rewards_list)

        is_main = (self.accelerator is None) or self.accelerator.is_main_process
        if log_sample and is_main:
            self._log_samples(completions, labels, rewards_list, g, step, cer_scores, judge_scores)

        # 组内优势标准化
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        grouped = rewards.view(b, g)
        adv     = (grouped - grouped.mean(dim=1, keepdim=True)) / grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        adv     = adv.view(-1)

        # 构建训练输入
        train_inputs = self.builder.build_train_inputs(prompt_inputs, completions, g)
        train_inputs = {
            k: (v.to(self.device) if torch.is_tensor(v) else v)
            for k, v in train_inputs.items()
        }

        raw_model   = self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
        model_dtype = getattr(raw_model, "dtype", None)
        if model_dtype is not None and "input_features" in train_inputs and torch.is_tensor(train_inputs["input_features"]):
            train_inputs["input_features"] = train_inputs["input_features"].to(dtype=model_dtype)

        labels_mask = train_inputs.pop("labels")
        train_inputs.pop("prompt_lens")

        # 策略梯度前向
        self.model.train()
        out      = self.model(**train_inputs)
        log_probs = F.log_softmax(out.logits, dim=-1)

        input_ids    = train_inputs["input_ids"]
        shifted_ids  = input_ids[:, 1:]
        shifted_logp = log_probs[:, :-1, :]
        token_logp   = shifted_logp.gather(2, shifted_ids.unsqueeze(-1)).squeeze(-1)
        comp_mask    = (labels_mask[:, 1:] != -100).float()

        seq_logp = (token_logp * comp_mask).sum(dim=1) / comp_mask.sum(dim=1).clamp(min=1)
        pg_loss  = -(adv * seq_logp).mean()

        # KL 散度惩罚
        kl_loss = torch.tensor(0.0, device=self.device)
        if self.ref_model is not None and self.args.kl_coef > 0:
            with torch.inference_mode():
                ref_out       = self.ref_model(**train_inputs)
                ref_log_probs = F.log_softmax(ref_out.logits, dim=-1)
                ref_token_logp = ref_log_probs[:, :-1, :].gather(2, shifted_ids.unsqueeze(-1)).squeeze(-1)
            kl      = ((token_logp - ref_token_logp) * comp_mask).sum(dim=1) / comp_mask.sum(dim=1).clamp(min=1)
            kl_loss = kl.mean()

        loss = (pg_loss + self.args.kl_coef * kl_loss) / self.args.grad_acc

        if self.accelerator:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        return {
            "loss":    float(loss.item() * self.args.grad_acc),
            "pg_loss": float(pg_loss.item()),
            "kl_loss": float(kl_loss.item()),
            "reward":  float(rewards.mean().item()),
            "cer":     float(sum(cer_scores)   / max(1, len(cer_scores))),
            "judge":   float(sum(judge_scores) / max(1, len(judge_scores))),
        }

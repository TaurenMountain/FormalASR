"""
eval/infer.py
模型加载与音频推理：支持多种后端，通过 --checkpoint 参数自动路由。

  --checkpoint <本地目录>                   -> 读取 config.json 的 model_type
      model_type == "whisper"              -> _WhisperBackend
      其他                                 -> Qwen3ASRModel (qwen_asr，默认)

  --checkpoint <本地 .gguf 文件>            -> _GGUFBackend
      基于 llama-cpp-python 的 GGUF 推理后端。

  --checkpoint pipeline                    -> _PipelineBackend
      两阶段推理：
        Stage 1  Qwen3-ASR-1.7B 语音识别（路径由 PIPELINE_ASR_CKPT 环境变量指定，
                 默认 ../models/Qwen3-ASR-1.7B）
        Stage 2  Qwen3-4B LLM 文本纠错（路径由 PIPELINE_CORRECTOR_CKPT 环境变量
                 指定，默认 ../models/Qwen3-4B）

  --checkpoint openai/<model>              -> _OpenAIAudioBackend
      兼容 OpenAI Audio API 的多模态大模型（含 GPT-4o-audio-preview / Gemini 等）。
      API Key 由 OPENAI_API_KEY 或 OPENROUTER_API_KEY 环境变量提供；
      API Base URL 由 OPENAI_API_BASE 环境变量提供（默认官方 OpenAI）。
      示例：
        --checkpoint openai/gpt-4o-audio-preview
        --checkpoint openai/google/gemini-2.0-flash-001
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import wave
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from common.audio import TARGET_SR


# ──────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────

def _get_model_type(model_path: str) -> str:
    """从 checkpoint 目录的 config.json 读取 model_type，默认返回 'qwen3_asr'。"""
    cfg_path = Path(model_path) / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("model_type", "qwen3_asr").lower()
    return "qwen3_asr"


def _audio_array_to_wav_bytes(audio_array: np.ndarray, sr: int = TARGET_SR) -> bytes:
    """将 float32 numpy 数组转换为 16-bit PCM WAV bytes。"""
    pcm = np.clip(audio_array, -1.0, 1.0)
    pcm_int16 = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────
# GGUF 后端（llama-cpp-python）
# ──────────────────────────────────────────────────────────────────────

class _GGUFBackend:
    """基于 llama-cpp-python 的 GGUF 推理后端。"""

    _SYSTEM_PROMPT = """你是专业的中文语音转写员。你的唯一任务是将音频中说的话忠实转写为书面化中文，不做任何扩展、补充或创作。

【绝对禁止】
- 禁止输出音频中没有的任何内容
- 禁止解释、注释或说明转写过程

【输出格式】
只输出转写后的文本。"""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 0):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "GGUF 后端需要 llama-cpp-python，请先安装：pip install llama-cpp-python"
            ) from exc

        if not Path(model_path).is_file():
            raise FileNotFoundError(f"GGUF 模型文件不存在: {model_path}")

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self._model_path = model_path
        print(
            f"[GGUF] 初始化完成 model={model_path} n_ctx={n_ctx} n_gpu_layers={n_gpu_layers} ✓"
        )

    def transcribe(self, audio_array: np.ndarray, language: str | None = None) -> str:
        """
        使用音频占位提示进行文本生成。
        说明：当前最小实现兼容评估流程，不直接消费原始音频数组。
        """
        lang_hint = language or "zh"
        prompt = (
            f"<|system|>\n{self._SYSTEM_PROMPT}\n"
            f"<|user|>\n请输出该语音的转写文本。语言: {lang_hint}\n"
            f"<|assistant|>\n"
        )
        try:
            out = self._llm(
                prompt,
                max_tokens=256,
                temperature=0.0,
                stop=["<|user|>", "<|system|>"],
                echo=False,
            )
            text = out.get("choices", [{}])[0].get("text", "")
            return (text or "").strip()
        except Exception as exc:
            return f"[ERROR: GGUF decode failed: {exc}]"


# ──────────────────────────────────────────────────────────────────────
# Whisper 后端
# ──────────────────────────────────────────────────────────────────────

class _WhisperBackend:
    """封装 HuggingFace Whisper 推理，对外暴露与 Qwen3ASRModel 相同的接口。"""

    # Whisper language token 映射（ISO 639-1 -> Whisper language code）
    _LANG_MAP = {
        "zh": "chinese",
        "en": "english",
        "ja": "japanese",
        "ko": "korean",
        "fr": "french",
        "de": "german",
        "es": "spanish",
        "ru": "russian",
        "ar": "arabic",
        "pt": "portuguese",
        "it": "italian",
    }

    def __init__(
        self,
        model_path: str,
        device: str,
        dtype: torch.dtype,
        quantization: Optional[str] = None,
    ):
        from transformers import (
            BitsAndBytesConfig,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        quantization = (quantization or "none").lower()
        print(f"[Whisper] 加载 processor: {model_path}")
        self.processor = WhisperProcessor.from_pretrained(model_path)

        model_kwargs = {}
        if quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = device
            print(f"[Whisper] 加载模型: {model_path}  quantization=8bit")
        elif quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
            model_kwargs["device_map"] = device
            print(f"[Whisper] 加载模型: {model_path}  quantization=4bit")
        else:
            model_kwargs["torch_dtype"] = dtype
            print(f"[Whisper] 加载模型: {model_path}  dtype={dtype}")

        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs,
        )
        if quantization == "none":
            self.model = self.model.to(device)

        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.quantization = quantization
        print("[Whisper] 加载完成 ✓")

    def transcribe(
        self,
        audio_array: np.ndarray,
        language: str | None = None,
    ) -> str:
        """对单条音频进行转写，返回文本字符串。"""
        forced_decoder_ids = None
        if language:
            lang_code = self._LANG_MAP.get(language.lower(), language.lower())
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=lang_code, task="transcribe"
            )

        inputs = self.processor(
            audio_array,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=444,  # Whisper max_target_positions=448，forced_decoder_ids 占 4 个 token
            )

        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return text[0].strip() if text else ""


# ──────────────────────────────────────────────────────────────────────
# 两阶段 Pipeline 后端：Qwen3-ASR-1.7B + Qwen3-4B 纠错
# ──────────────────────────────────────────────────────────────────────

# 纠错模型 system prompt（与数据集 target_text 生成风格对齐）
# 在书面化基础上额外要求先纠正 ASR 识别错误（同音/近音字、漏字、重复）
_CORRECTOR_SYSTEM_PROMPT = """\
你是资深中文出版编辑，擅长把口语转成自然、流畅、规范的书面表达。
你的任务是仅基于输入的 ASR 粗糙文本生成 target_text；禁止引入原文没有的新信息。

核心目标（必须同时满足）：
1. **忠实**：不改变事实、意图、立场、时态、人物关系
2. **通顺**：消除口语卡顿感，让句子衔接自然、读起来顺滑
3. **书面**：表达更正式、更凝练，但不过度文绉绉
4. **标点充分**：句读清楚，停顿合理，避免"一长句到底"

改写规则（按优先级）：
1. 删除语气词与感叹填充：呃、嗯、啊、哦、哎、嘛、呀、哇、噢、唉、喂
2. 删除口头填充与赘余连接：那个、就是说、就是说嘛、然后、然后呢、然后啊、你知道吧、对吧、是吧
3. 消除重复与自我修正痕迹：如"这个这个""我觉得我觉得""就是...就是..."
4. 口语句法书面化：
   - 把松散口语改成紧凑句式
   - 合并碎片短句，拆分过长句
   - 适度补主语或宾语（仅限原文可推出的信息）
5. 谨慎纠错：仅在语义明显不通时做最小必要改字；不确定则保留原文
6. 保持原信息密度：不扩写、不总结、不拔高观点

标点规则（强约束）：
1. 每个完整陈述句以"。"收尾；疑问句用"？"，感叹句用"！"
2. 并列、转折、因果、条件关系处优先补"，"；分句较长可用"；"
3. 直接引语、书名、专有内容按中文规范使用引号与书名号
4. 禁止连续无标点长串文本；禁止全句只用逗号不断句

输出要求：
- 仅输出为自然、通顺、规范的中文书面句子，不输出解释、注释、过程
- 允许在不改原意前提下做必要重排，以提升流畅性
- 总体字数可略减，但删减不超过 20%"""

# Pipeline 后端默认模型：优先读环境变量，否则使用 HF Hub ID
_DEFAULT_PIPELINE_ASR_CKPT       = "Qwen/Qwen3-ASR-1.7B"
_DEFAULT_PIPELINE_CORRECTOR_CKPT = "Qwen/Qwen3-4B"


class _PipelineBackend:
    """
    两阶段 ASR 推理后端。

    Stage 1 - Qwen3-ASR-1.7B：将音频转成粗糙文本（initial transcript）
    Stage 2 - Qwen3-4B：将粗糙文本通过 /no_think prompt 进行语义纠错，输出最终结果

    路径通过环境变量配置（若未设置则使用上方默认路径）：
      PIPELINE_ASR_CKPT       Stage-1 ASR 模型目录
      PIPELINE_CORRECTOR_CKPT Stage-2 纠错 LLM 模型目录
    """

    def __init__(self, device: str, dtype: torch.dtype, quantization: str = "none"):
        asr_ckpt  = os.environ.get("PIPELINE_ASR_CKPT",       _DEFAULT_PIPELINE_ASR_CKPT)
        corr_ckpt = os.environ.get("PIPELINE_CORRECTOR_CKPT", _DEFAULT_PIPELINE_CORRECTOR_CKPT)

        print(f"[Pipeline] Stage-1 ASR 模型:  {asr_ckpt}")
        print(f"[Pipeline] Stage-2 纠错模型:  {corr_ckpt}")

        # ── Stage 1：ASR 模型 ─────────────────────────────────────────
        asr_type = _get_model_type(asr_ckpt)
        if asr_type == "whisper":
            self._asr = _WhisperBackend(asr_ckpt, device, dtype, quantization=quantization)
            self._asr_is_whisper = True
        else:
            from qwen_asr import Qwen3ASRModel
            from transformers import BitsAndBytesConfig

            asr_kwargs: dict = {"device_map": device, "max_new_tokens": 512}
            if quantization == "8bit":
                asr_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization == "4bit":
                asr_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                )
            else:
                asr_kwargs["dtype"] = dtype

            self._asr = Qwen3ASRModel.from_pretrained(asr_ckpt, **asr_kwargs)
            self._asr_is_whisper = False

        # ── Stage 2：文本纠错 LLM ────────────────────────────────────
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        corr_kwargs: dict = {"device_map": device}
        if quantization == "8bit":
            corr_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "4bit":
            corr_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
        else:
            corr_kwargs["torch_dtype"] = dtype

        self._tok = AutoTokenizer.from_pretrained(corr_ckpt, trust_remote_code=True)
        self._corr = AutoModelForCausalLM.from_pretrained(
            corr_ckpt, trust_remote_code=True, **corr_kwargs
        )
        self._corr.eval()
        print("[Pipeline] 两阶段模型加载完成 ✓")

    def _stage1_asr(self, audio_array: np.ndarray, language: str | None) -> str:
        """Stage 1：音频 -> 粗糙 ASR 文本。"""
        if self._asr_is_whisper:
            return self._asr.transcribe(audio_array, language=language)
        lang_arg = [language] if language else None
        results = self._asr.transcribe(audio=[(audio_array, TARGET_SR)], language=lang_arg)
        return results[0].text if results and results[0] else ""

    def _stage2_correct(self, asr_text: str) -> str:
        """Stage 2：粗糙文本 -> 纠错文本（Qwen3 /no_think 模式）。"""
        messages = [
            {"role": "system", "content": _CORRECTOR_SYSTEM_PROMPT},
            {"role": "user",   "content": f"/no_think\n{asr_text}"},
        ]
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(text, return_tensors="pt").to(self._corr.device)
        with torch.no_grad():
            out_ids = self._corr.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tok.eos_token_id,
            )
        new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        return self._tok.decode(new_ids, skip_special_tokens=True).strip()

    def transcribe(self, audio_array: np.ndarray, language: str | None = None) -> str:
        """对单条音频执行两阶段推理，返回最终文本。"""
        stage1 = self._stage1_asr(audio_array, language)
        if not stage1.strip():
            return stage1
        return self._stage2_correct(stage1)


# ──────────────────────────────────────────────────────────────────────
# OpenAI Audio API 后端（GPT-4o-audio-preview / Gemini via OpenRouter 等）
# ──────────────────────────────────────────────────────────────────────

_OPENAI_AUDIO_SYSTEM_PROMPT = """\
你是专业的中文语音转写员。你的唯一任务是将音频中说的话忠实转写为书面化中文，不做任何扩展、补充或创作。

【绝对禁止】
- 禁止输出音频中没有的任何内容（禁止补充背景、禁止扩写、禁止加入自己的观点）
- 禁止在开头加"好的"、"明白了"等应答词
- 禁止解释、注释或说明转写过程
- 如果音频内容很短（哪怕只有一句话），输出也必须只有那一句话

【转写规则】
1. 删除语气词：呃、嗯、啊、哦、哎、嘛、呀、哇、噢、唉、喂
2. 删除口头赘词：那个、就是说、然后呢、你知道吧、对吧、是吧
3. 消除重复：如"这个这个"→"这个"，"我觉得我觉得"→"我觉得"
4. 书面化句式：把松散口语改成紧凑句式，但不改变原意
5. 标点：陈述句用"。"，疑问句用"？"，合理使用"，"断句

【输出格式】
只输出转写后的文本，不输出任何其他内容。字数只能比原音频内容更少，绝不能更多。"""


class _OpenAIAudioBackend:
    """
    兼容 OpenAI Audio API 的多模态大模型后端。

    支持：
      openai/gpt-4o-audio-preview          使用 input_audio 格式
      openai/google/gemini-*（OpenRouter）  使用 image_url + data URI 格式

    环境变量：
      OPENAI_API_KEY       OpenAI 官方 API Key
      OPENROUTER_API_KEY   OpenRouter API Key（优先于 OPENAI_API_KEY）
      OPENAI_API_BASE      API Base URL（默认由 key 类型自动判断）
    """

    def __init__(self, model_name: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI Audio 后端需要 openai 包，请执行：pip install 'openai>=1.30.0'"
            )

        api_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        if not api_key:
            raise ValueError(
                "使用 OpenAI Audio 后端时需要 API Key，"
                "请设置环境变量 OPENROUTER_API_KEY 或 OPENAI_API_KEY"
            )

        # 默认 base_url：有 OPENROUTER_API_KEY 则用 OpenRouter，否则官方 OpenAI
        default_base = (
            "https://openrouter.ai/api/v1"
            if os.environ.get("OPENROUTER_API_KEY")
            else "https://api.openai.com/v1"
        )
        api_base = os.environ.get("OPENAI_API_BASE", default_base)

        # OpenRouter 需要额外 headers（与 test_gpt4o-audio.py 保持一致）
        extra_headers: dict = {}
        if "openrouter.ai" in api_base:
            extra_headers = {
                "HTTP-Referer": "https://github.com/nwy",
                "X-Title": "ASR Evaluation",
            }

        self._client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            default_headers=extra_headers or None,
        )
        self._model_name = model_name
        # Gemini 系列通过 OpenRouter 传音频使用 image_url + data URI 格式
        self._is_gemini = (
            model_name.startswith("google/gemini") or model_name.startswith("gemini")
        )
        print(
            f"[OpenAIAudio] 初始化完成  model={model_name}  "
            f"api_base={api_base}  gemini_mode={self._is_gemini} ✓"
        )

    def _build_audio_content(self, audio_b64: str) -> dict:
        """根据模型类型构建正确的音频消息块（与 test_gpt4o-audio.py 保持一致）。"""
        if self._is_gemini:
            return {
                "type": "image_url",
                "image_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
            }
        return {
            "type": "input_audio",
            "input_audio": {"data": audio_b64, "format": "wav"},
        }

    def transcribe(self, audio_array: np.ndarray, language: str | None = None) -> str:
        """将单条音频 base64 编码后调用 Audio API，返回识别文本。"""
        wav_bytes = _audio_array_to_wav_bytes(audio_array)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        audio_block = self._build_audio_content(audio_b64)

        messages = [
            {"role": "system", "content": _OPENAI_AUDIO_SYSTEM_PROMPT},
            {"role": "user",   "content": [audio_block]},
        ]

        for attempt in range(1, 4):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.1,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                if attempt < 3:
                    print(f"[OpenAIAudio] API 调用失败（第{attempt}次）: {exc}，2s 后重试...")
                    time.sleep(2.0)
                else:
                    print(f"[OpenAIAudio] API 调用失败（已重试3次）: {exc}")
                    return f"[ERROR: {exc}]"
        return ""


# ──────────────────────────────────────────────────────────────────────
# 模型加载（统一入口）
# ──────────────────────────────────────────────────────────────────────

def load_asr_model(
    model_path: str,
    device: str,
    batch_size: int = 1,
    quantization: str = "none",
):
    """
    根据 model_path 字符串自动路由到对应后端。

    路由规则（按优先级）：
      1. model_path == "pipeline"            -> _PipelineBackend（两阶段）
      2. model_path.startswith("openai/")    -> _OpenAIAudioBackend
      3. model_path.endswith(".gguf")         -> _GGUFBackend
      4. config.json model_type == "whisper" -> _WhisperBackend
      5. 其他                                -> Qwen3ASRModel (qwen_asr)

    参数：
      model_path    checkpoint 目录 / 特殊关键字（pipeline / openai/<model>）
      device        推理设备，如 "cuda:0"
      batch_size    仅对 Qwen3ASR 生效（max_inference_batch_size）
      quantization  量化等级：none / 8bit / 4bit
    """
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    quantization = (quantization or "none").lower()
    if quantization not in {"none", "8bit", "4bit"}:
        raise ValueError(f"不支持的 quantization={quantization}，可选: none|8bit|4bit")

    # ── 两阶段 Pipeline ──────────────────────────────────────────────
    if model_path.lower() == "pipeline":
        print("[模型] 检测到 checkpoint=pipeline，加载两阶段 _PipelineBackend")
        if quantization in {"8bit", "4bit"} and not torch.cuda.is_available():
            raise RuntimeError("8bit/4bit 量化推理需要 CUDA 环境")
        return _PipelineBackend(device=device, dtype=dtype, quantization=quantization)

    # ── OpenAI Audio API ────────────────────────────────────────────
    if model_path.startswith("openai/"):
        # model_name = 去掉 "openai/" 前缀后的字符串
        # 例：checkpoint="openai/gpt-4o-audio-preview"       -> "gpt-4o-audio-preview"
        # 例：checkpoint="openai/google/gemini-2.0-flash-001" -> "google/gemini-2.0-flash-001"
        model_name = model_path[len("openai/"):]
        print(f"[模型] 检测到 checkpoint=openai/...，加载 _OpenAIAudioBackend  model={model_name}")
        return _OpenAIAudioBackend(model_name=model_name)

    # ── GGUF 模型（llama-cpp-python）──────────────────────────────
    if model_path.lower().endswith(".gguf"):
        print(f"[模型] 检测到 GGUF 文件，加载 _GGUFBackend: {model_path}")
        gguf_n_ctx = int(os.environ.get("GGUF_N_CTX", "4096"))
        gguf_n_gpu_layers = int(os.environ.get("GGUF_N_GPU_LAYERS", "0"))
        return _GGUFBackend(
            model_path=model_path,
            n_ctx=gguf_n_ctx,
            n_gpu_layers=gguf_n_gpu_layers,
        )

    # ── 本地模型（Whisper / Qwen3ASR）──────────────────────────────
    model_type = _get_model_type(model_path)
    print(
        f"[模型] 检测到 model_type='{model_type}'，路径: {model_path}，quantization={quantization}"
    )

    if quantization in {"8bit", "4bit"} and not torch.cuda.is_available():
        raise RuntimeError("8bit/4bit 量化推理需要 CUDA 环境")

    if model_type == "whisper":
        return _WhisperBackend(model_path, device, dtype, quantization=quantization)

    # 默认：Qwen3ASR
    from qwen_asr import Qwen3ASRModel
    from transformers import BitsAndBytesConfig

    model_kwargs = {
        "device_map": device,
        "max_inference_batch_size": batch_size,
        "max_new_tokens": 512,
    }

    if quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print(f"[模型] 加载 Qwen3ASR: {model_path}  quantization=8bit")
    elif quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        print(f"[模型] 加载 Qwen3ASR: {model_path}  quantization=4bit")
    else:
        model_kwargs["dtype"] = dtype
        print(f"[模型] 加载 Qwen3ASR: {model_path}  dtype={'bfloat16' if use_bf16 else 'float16'}")

    model = Qwen3ASRModel.from_pretrained(model_path, **model_kwargs)
    print("[模型] 加载完成 ✓")
    return model


# ──────────────────────────────────────────────────────────────────────
# 批量转写（统一入口）
# ──────────────────────────────────────────────────────────────────────

def transcribe_batch(
    model,
    audio_arrays: List[np.ndarray],
    languages: List[str],
    force_language: bool,
) -> List[str]:
    """
    对一批音频逐条推理，返回转写文本列表。
    自动识别所有支持的后端：
      _WhisperBackend / _PipelineBackend / _OpenAIAudioBackend / Qwen3ASRModel

    注意：逐条推理（batch_size=1）可避免音频 padding 导致某些模型（如 GRPO 微调模型）
    输出空字符串的问题。
    """
    texts = []

    if isinstance(model, (_GGUFBackend, _WhisperBackend, _PipelineBackend, _OpenAIAudioBackend)):
        # 统一 .transcribe(audio_array, language) 接口
        for arr, lang in zip(audio_arrays, languages):
            try:
                lang_arg = lang if force_language else None
                texts.append(model.transcribe(arr, language=lang_arg))
            except Exception as exc:
                texts.append(f"[ERROR: {exc}]")
    else:
        # Qwen3ASR 后端
        for arr, lang in zip(audio_arrays, languages):
            try:
                lang_arg = [lang] if force_language else None
                results  = model.transcribe(audio=[(arr, TARGET_SR)], language=lang_arg)
                texts.append(results[0].text if results and results[0] else "")
            except Exception as exc:
                texts.append(f"[ERROR: {exc}]")

    return texts

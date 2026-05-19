#!/usr/bin/env bash
# run_eval.sh  —  评估启动脚本
#
# 数据集直接从 HuggingFace Hub 下载，无需本地 JSONL 文件。
#
# 用法一：标准 Qwen3-ASR / FormalASR checkpoint
#   bash run_eval.sh
#   CUDA_VISIBLE_DEVICES=0 CKPT=TaurenMountain/FormalASR-0.6B bash run_eval.sh
#   CUDA_VISIBLE_DEVICES=0 CKPT=checkpoints/formalASR-sft/checkpoint-1000 bash run_eval.sh
#
# 用法二：两阶段 Pipeline（Qwen3-ASR-1.7B -> Qwen3-4B 纠错）
#   CKPT=pipeline bash run_eval.sh
#   可通过环境变量自定义两阶段模型路径（默认读 HF Hub 上的 Qwen 官方模型）：
#   CKPT=pipeline \
#     PIPELINE_ASR_CKPT=Qwen/Qwen3-ASR-1.7B \
#     PIPELINE_CORRECTOR_CKPT=Qwen/Qwen3-4B \
#     bash run_eval.sh
#
# 用法三：多模态大模型 GPT-4o-audio-preview（OpenAI Audio API）
#   CKPT=openai/gpt-4o-audio-preview OPENAI_API_KEY=sk-xxx bash run_eval.sh
#
# 用法四：Gemini via OpenRouter
#   CKPT=openai/google/gemini-2.0-flash-001 \
#     OPENROUTER_API_KEY=sk-or-xxx \
#     OPENAI_API_BASE=https://openrouter.ai/api/v1 \
#     bash run_eval.sh
#
# 多数据集：DATASETS=wenetspeech,speechio bash run_eval.sh
# 从断点恢复：RESUME=1 bash run_eval.sh（默认即为 1；RESUME=0 从头覆盖）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CKPT=${CKPT:-"TaurenMountain/FormalASR-1.7B"}
DATASETS=${DATASETS:-"wenetspeech"}
QUANTIZATION=${QUANTIZATION:-none}
RESUME=${RESUME:-1}

cd "${SCRIPT_DIR}"

python -m eval.run \
    --checkpoint      "${CKPT}" \
    --dataset_names   "${DATASETS}" \
    --output_dir      "eval_results/${DATASETS//,/_}_$(basename "${CKPT}")" \
    --batch_size      4 \
    --max_samples     -1 \
    --device          "cuda:0" \
    --quantization    "${QUANTIZATION}" \
    --bert_device     "cpu" \
    --resume          "${RESUME}"

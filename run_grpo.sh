#!/usr/bin/env bash
# run_grpo.sh  —  GRPO 强化学习微调启动脚本
#
# 数据集直接从 HuggingFace Hub 下载，无需本地 JSONL 文件。
#
# 单卡（默认 GPU 0）：
#   bash run_grpo.sh
#
# 多卡示例：
#   CUDA_VISIBLE_DEVICES=0,1 bash run_grpo.sh
#
# 自定义 checkpoint：
#   CKPT=checkpoints/formalASR-sft/checkpoint-1000 bash run_grpo.sh
#
# 自定义模型 ID：
#   CKPT=TaurenMountain/FormalASR-0.6B OUTPUT_DIR=checkpoints/formalASR-0.6b-grpo bash run_grpo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 可自定义的变量 ────────────────────────────────────────────────────
CKPT="${CKPT:-TaurenMountain/FormalASR-1.7B}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/checkpoints/formalASR-grpo}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"   # -1 表示使用整个数据集
JUDGE_WEIGHT="${JUDGE_WEIGHT:-0}"
# ─────────────────────────────────────────────────────────────���───────

# ── GPU / 进程数配置 ─────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [ -z "${NUM_PROCESSES:-}" ]; then
    NUM_PROCESSES=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
fi

echo "[run_grpo] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  NUM_PROCESSES=${NUM_PROCESSES}"

cd "${SCRIPT_DIR}"

if [ "${NUM_PROCESSES}" -le 1 ]; then
    LAUNCHER="python -m"
else
    LAUNCHER="accelerate launch --num_processes ${NUM_PROCESSES} --multi_gpu -m"
fi

# 断点续传：若 OUTPUT_DIR 下存在 latest_checkpoint.txt 则自动恢复
RESUME="${RESUME:-latest}"

# shellcheck disable=SC2086
${LAUNCHER} grpo.train \
    --model_path              "${CKPT}" \
    --dataset_name            "wenetspeech" \
    --output_dir              "${OUTPUT_DIR}" \
    --epochs                  1 \
    --batch_size              1 \
    --num_generations         4 \
    --grad_acc                4 \
    --lr                      5e-7 \
    --kl_coef                 0.01 \
    --temperature             0.9 \
    --top_p                   0.95 \
    --max_completion_length   80 \
    --log_steps               10 \
    --save_steps              100 \
    --save_total_limit        3 \
    --max_samples             "${MAX_SAMPLES}" \
    --resume_from_checkpoint  "${RESUME}" \
    --reward_judge_weight     "${JUDGE_WEIGHT}"

#!/usr/bin/env bash
# run_sft.sh  —  SFT 全量微调启动脚本
#
# 数据集直接从 HuggingFace Hub 下载，无需本地 JSONL 文件。
#
# 用法示例：
#
#   【默认（单 GPU）】
#   bash run_sft.sh
#
#   【多卡 DDP】
#   CUDA_VISIBLE_DEVICES=0,1 bash run_sft.sh
#
#   【自定义数据集】
#   DATASET_NAME=speechio bash run_sft.sh
#
#   【自定义模型】
#   MODEL_PATH=TaurenMountain/FormalASR-0.6B bash run_sft.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 统计可见 GPU 数量（逗号分隔的设备 ID 个数）
_gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c '[0-9]')

if [ "$_gpu_count" -gt 1 ]; then
    echo "[run_sft] 检测到 ${_gpu_count} 张 GPU，使用 torchrun DDP 启动..."
    LAUNCHER="torchrun --nproc_per_node=${_gpu_count} --master_port=29500 -m sft.train"
else
    echo "[run_sft] 单卡模式，使用 python 启动..."
    LAUNCHER="python -m sft.train"
fi

# ── 可自定义的变量（可在调用前用环境变量覆盖） ──────────────────────────
MODEL_PATH="${MODEL_PATH:-TaurenMountain/FormalASR-1.7B}"
DATASET_NAME="${DATASET_NAME:-TaurenMountain/WenetSpeech-Formal}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/formalASR-sft}"
# ────────────────────────────────────────────────────────────────────────

cd "${SCRIPT_DIR}"

$LAUNCHER \
    --model_path       "${MODEL_PATH}" \
    --dataset_name     "${DATASET_NAME}" \
    --output_dir       "${OUTPUT_DIR}" \
    --epochs           1 \
    --batch_size       4 \
    --grad_acc         8 \
    --lr               2e-5 \
    --save_steps       200 \
    --save_total_limit 3 \
    --log_steps        10

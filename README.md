# FormalASR: End-to-End Spoken Chinese to Formal Text

<p align="center">
  <a href="https://huggingface.co/TaurenMountain/FormalASR-0.6B"><img src="https://img.shields.io/badge/🤗 FormalASR--0.6B-Model-yellow" /></a>
  <a href="https://huggingface.co/TaurenMountain/FormalASR-1.7B"><img src="https://img.shields.io/badge/🤗 FormalASR--1.7B-Model-yellow" /></a>
  <a href="https://huggingface.co/datasets/TaurenMountain/WenetSpeech-Formal"><img src="https://img.shields.io/badge/🤗 WenetSpeech--Formal-Dataset-orange" /></a>
  <a href="https://huggingface.co/datasets/TaurenMountain/Speechio-Formal"><img src="https://img.shields.io/badge/🤗 Speechio--Formal-Dataset-orange" /></a>
  <a href="#"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.8+-green.svg" /></a>
  <a href="https://www.yijiahe.com/"><img src="https://img.shields.io/badge/Released%20by-亿嘉和-red" /></a>
</p>

**FormalASR** 是一组基于 [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) 微调的紧凑型端到端中文语音识别模型，专为**书面体转写**设计：直接从语音输入输出规范的书面文本，自动完成去除口语填充词、句式和标点规范化，**无需额外的 LLM 后处理步骤**，适合设备端部署。

> **口语输入** → 这个上上一像这个进料这一块一定要把关哪怕说是价钱贵点儿然后咱进点儿好料咱能保障咱
>
> **FormalASR 输出** → 进料这一块一定要把关，哪怕价钱贵点，我们也能进点好料，以保障质量。

---

## 🔥 新闻

- **2026-05**: 发布 FormalASR-0.6B、FormalASR-1.7B 模型及 WenetSpeech-Formal、Speechio-Formal 数据集！
- **2026-05**: 论文 [*FormalASR: End-to-End Spoken Chinese to Formal Text*](https://arxiv.org/abs/2026.XXXXX) 上传 arXiv。

---

## 📖 论文摘要

ASR 系统通常针对逐字转写进行优化，保留了口语中的填充词、重复和非正式句式，不适合文档生成等下游应用。常见的两阶段 ASR+LLM 方案会带来延迟和内存开销，难以在设备端部署。

本文提出 **FormalASR**，两个紧凑型端到端模型（0.6B 和 1.7B），直接将普通话语音转写为规范书面文本。为此，我们构建了 **WenetSpeech-Formal**（969K 训练样本）和 **Speechio-Formal**（43K 跨域测试样本）两个大规模书面体 ASR 数据集，通过 LLM 改写和质量过滤生成。在 WenetSpeech 和 Speechio 上的实验表明，FormalASR 在 CER、ROUGE-L 和 BERTScore 上均显著优于逐字转写基线。完整论文见 [arXiv](https://arxiv.org/abs/2026.XXXXX)。

---

## 📂 数据集

| 数据集 | 样本量 | 音频来源 | 标注方式 | 用途 | HuggingFace |
|--------|--------|----------|----------|------|-------------|
| **WenetSpeech-Formal** (train) | 969,201 | WenetSpeech | DeepSeek-V3.2 书面化改写 | SFT 训练 | [🤗 TaurenMountain/WenetSpeech-Formal](https://huggingface.co/datasets/TaurenMountain/WenetSpeech-Formal) |
| **WenetSpeech-Formal** (test) | 31,932 | WenetSpeech | DeepSeek-V3.2 书面化改写 | 域内评估 | 同上 |
| **Speechio-Formal** (test) | 43,178 | Speechio (ZH00000–ZH00026) | DeepSeek-V3.2 书面化改写 | 跨域评估 | [🤗 TaurenMountain/Speechio-Formal](https://huggingface.co/datasets/TaurenMountain/Speechio-Formal) |

### 数据构建流程

1. **逐字转写收集**：使用 WenetSpeech / Speechio 的原始音频及其逐字口语转写
2. **LLM 书面化改写**：调用 **DeepSeek-V3.2** 将口语文本改写为规范书面体（去填充词、重组句式、规范标点）
3. **质量过滤**：过滤嵌入相似度过低（语义漂移）、编辑距离异常（无实质改写或幻觉）的样本

### 数据示例

| 原始口语 (`original_text`) | 书面体 (`target_text`) |
|---------------------------|----------------------|
| 把这个呃增加的这个利润 | 把这个增加的利润。 |
| 对全美国全球影响影响不大 | 对美国全球影响不大。 |
| 但是我想这里这当中就是如果如果一定要那个挑一点儿什么的话 | 但是，如果一定要从中挑出一点什么的话。 |

---

## 📦 模型

| 模型 | 参数量 | 训练方式 | HuggingFace |
|------|--------|----------|-------------|
| **FormalASR-0.6B** | 0.6B | SFT 全量微调 | [🤗 TaurenMountain/FormalASR-0.6B](https://huggingface.co/TaurenMountain/FormalASR-0.6B) |
| **FormalASR-1.7B** | 1.7B | SFT 全量微调 | [🤗 TaurenMountain/FormalASR-1.7B](https://huggingface.co/TaurenMountain/FormalASR-1.7B) |

---

## 📊 评估结果

### 域内：WenetSpeech-Formal

| 模型 | CER ↓ | ROUGE-L ↑ | BERTScore F1 ↑ |
|------|--------|-----------|----------------|
| Qwen3-ASR-0.6B（基线） | 0.2581 | 0.8463 | 0.9198 |
| **FormalASR-0.6B** | **0.1770** | **0.8769** | **0.9359** |
| Qwen3-ASR-1.7B（基线） | 0.2460 | 0.8571 | 0.9268 |
| **FormalASR-1.7B** | **0.1606** | **0.8896** | **0.9439** |
| Whisper large-v3 | 0.3631 | 0.7393 | 0.8538 |

### 跨域：Speechio-Formal（27 个子集）

| 模型 | CER ↓ | ROUGE-L ↑ | BERTScore F1 ↑ |
|------|--------|-----------|----------------|
| Qwen3-ASR-0.6B（基线） | 0.2252 | 0.8701 | 0.9343 |
| **FormalASR-0.6B** | **0.1603** | **0.8948** | **0.9481** |
| Qwen3-ASR-1.7B（基线） | 0.2393 | 0.8510 | 0.9108 |
| **FormalASR-1.7B** | **0.1499** | **0.9029** | **0.9533** |
| Whisper large-v3 | 0.3302 | 0.7643 | 0.8795 |

> **评估指标说明**：CER（字符错误率，↓越低越好）；ROUGE-L（最长公共子序列 F1，↑越高越好）；BERTScore F1（语义相似度，↑越高越好）

### 推理效率

FormalASR-1.7B 相比 Qwen3-ASR-1.7B，在 WenetSpeech-Formal 上**平均输出 token 减少 22.8%**（18.5→14.3），对于较长句子（40-49 token 区间）延迟减少约 388ms，书面化转写省去 filler 词使解码步骤更少。

---

## 🚀 快速开始

### 安装

```bash
pip install -U qwen-asr
```

### HuggingFace 推理

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "TaurenMountain/FormalASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_new_tokens=512,
)

results = model.transcribe(
    audio="your_audio.wav",
    language="Chinese",
)

print(results[0].text)
```

### 魔搭社区（ModelScope）

```python
import torch
from modelscope import snapshot_download
from qwen_asr import Qwen3ASRModel

model_dir = snapshot_download("TaurenMountain/FormalASR-1.7B")

model = Qwen3ASRModel.from_pretrained(
    model_dir,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_new_tokens=512,
)

results = model.transcribe(
    audio="your_audio.wav",
    language="Chinese",
)

print(results[0].text)
```

---

## 🏋️ 训练

### 安装依赖

```bash
pip install -U qwen-asr transformers datasets accelerate torch
```

### SFT 全量微调

数据集自动从 HuggingFace Hub 下载，无需手动准备本地数据。

```bash
# 单卡
bash run_sft.sh

# 多 GPU
CUDA_VISIBLE_DEVICES=0,1 \
OUTPUT_DIR=checkpoints/formalASR-1.7b-sft \
bash run_sft.sh

# 自定义基础模型
MODEL_PATH=TaurenMountain/FormalASR-0.6B bash run_sft.sh

# 从头训练（使用原始 Qwen3-ASR）
MODEL_PATH=Qwen/Qwen3-ASR-1.7B bash run_sft.sh
```

### GRPO 强化学习微调（实验性）

```bash
bash run_grpo.sh

# 基于 SFT checkpoint 继续 GRPO 训练
CKPT=checkpoints/formalASR-1.7b-sft/checkpoint-1000 bash run_grpo.sh
```

### 评估

```bash
# 评估 HuggingFace 上的模型
bash run_eval.sh

# 评估本地 checkpoint
CKPT=checkpoints/formalASR-1.7b-sft/checkpoint-1000 bash run_eval.sh

# 或使用 Python 命令自定义参数：
python -m eval.run \
    --checkpoint TaurenMountain/FormalASR-1.7B \
    --dataset_names wenetspeech,speechio \
    --output_dir eval_results/ \
    --batch_size 4
```

---

## 📁 项目结构

```
FormalASR/
├── run_sft.sh              # SFT 训练启动脚本
├── run_grpo.sh             # GRPO 训练启动脚本（实验性）
├── run_eval.sh             # 评估启动脚本
├── common/                 # 公共工具
│   ├── audio.py            # 音频解码、HF Dataset 加载（含数据集 ID 映射）
│   ├── metrics.py          # CER / WER / ROUGE-L / BERTScore
│   └── model_utils.py      # patch_outer_forward（thinker 路由）
├── sft/                    # SFT 监督全量微调
│   ├── data.py             # 从 HF Dataset 构建训练记录 + DataCollator
│   ├── model.py            # Trainer / Callback 扩展
│   └── train.py            # 训练入口（支持多数据集混合）
├── grpo/                   # GRPO 强化学习微调（实验性）
│   ├── reward.py           # CER 奖励 + LLM judge 奖励
│   ├── trainer.py          # MinimalGRPOTrainer
│   └── train.py            # 训练入口
└── eval/                   # 评估
    ├── data.py             # 指标重计算工具
    ├── infer.py            # 多后端推理（Qwen3ASR / Whisper / Pipeline / OpenAI Audio）
    └── run.py              # 评估入口（支持多数据集）
```

---


## 📄 Citation

如果本项目对你的研究有帮助，请引用：

```bibtex
@article{ning2026formalasr,
  title   = {FormalASR: End-to-End Spoken Chinese to Formal Text},
  author  = {Ning, Wanyi and Guo, Yinshang and Qian, Haitao and Cheng, Jiyuan and Feng, Weiyuan and Zhang, Yufei},
  journal = {arXiv preprint arXiv:2026.XXXXX},
  year    = {2026},
  url     = {https://arxiv.org/abs/2026.XXXXX}
}
```

---

## 🙏 致谢

- 基础模型：[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- 训练数据来源：[WenetSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech)、[Speechio](https://github.com/SpeechColab/Leaderboard)
- LLM 书面化改写：[DeepSeek-V3.2](https://www.deepseek.com/)

---

<p align="center">Made with ❤️ by <a href="https://www.yijiahe.com/">亿嘉和技术股份有限公司</a></p>

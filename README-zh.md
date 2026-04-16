<div align="center">

# 记忆是否需要图？长期对话记忆的统一框架与实证分析

<p>
  <a href="https://arxiv.org/abs/2601.01280v2"><img src="https://img.shields.io/badge/arXiv-2601.01280-B31B1B?style=for-the-badge&logo=arxiv&logoColor=red" alt="arXiv Paper"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/🇺🇸English-1a1a2e?style=for-the-badge"></a>
</p>

<p align="center">
  <img src="./README.assets/unified_framework.png" style="max-width: min(100%, 1000px); height: auto;" alt="统一框架图示">
</p>

</div>

UnifiedMem 是一个面向长期对话记忆研究的统一框架，支持 flat 和 graph 两类方法，并围绕四个阶段组织流程：

- `index`
- `retrieve`
- `qa`
- `qa_eval`

## 🎉 News

- **2026.04.06** Our [**UnifiedMem**](https://arxiv.org/abs/2601.01280v3) is accepted by ACL 2026 Main Conference!

## Overview

本仓库实现了一个面向长期对话记忆的统一框架，覆盖 flat 与 graph 两类方法，并将完整流程组织为索引、检索、问答生成和问答评测四个阶段。

核心能力包括：

- 统一支持 flat 和 graph 两类记忆流程
- 支持摘要、关键词、用户事实等结构化记忆提取
- 支持为 `index`、`qa`、`qa_eval` 分别配置 LLM
- 检索与索引共享一套 embedding 配置
- 提供 LongMemEval 与 HaluMem 的评测流程

## 🚀 Quick Start

### 1. 安装依赖

```bash
conda create -n unifiedmem python=3.11 -y
conda activate unifiedmem
pip install -r requirements.txt
```

### 2. 创建根目录 `.env`

```bash
cp .env.example .env
```

最小配置：

```dotenv
OPENAI_API_KEY=""
OPENAI_BASE_URL="http://localhost:8001/v1"
LLM_MODEL="gpt-4o-mini"
EMBEDDING_MODEL="contriever"
```

如果不同阶段想使用不同 LLM，可以额外配置：

```dotenv
INDEX_API_KEY=""
INDEX_BASE_URL=""
INDEX_LLM_MODEL=""

QA_API_KEY=""
QA_BASE_URL=""
QA_LLM_MODEL=""

QA_EVAL_API_KEY=""
QA_EVAL_BASE_URL=""
QA_EVAL_LLM_MODEL=""
```

当阶段变量为空时，程序会自动回退到 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `LLM_MODEL`。

### 3. 准备数据

先将数据集放到下面的位置：

- LongMemEval: `data/longmemeval-cleaned/`
- HaluMem: `data/HaluMem/`

如果你是从 LongMemEval 原始清洗版数据开始，先放置这些文件：

- `data/longmemeval-cleaned/longmemeval_s_cleaned.json`
- `data/longmemeval-cleaned/longmemeval_oracle.json`
- `data/HaluMem/HaluMem-Medium.jsonl`

对于 LongMemEval，在做索引和检索之前，需要先运行一次去重预处理：

```bash
python data_preprocessing/lme_deduplicate.py
```

完成预处理之后，仓库中的流程通常使用下面这些文件：

- `data/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json`
- `data/longmemeval-cleaned/longmemeval_oracle_deduplicate.json`
- `data/HaluMem/HaluMem-Medium.jsonl`

### 4. ▶️ 运行流程

LongMemEval flat 检索：

```bash
./scripts/flat_lme_build_index.sh

python -m src.flat.lme_run_retrieval \
  --out_dir results/flat_lme
```

LongMemEval flat 流程默认会读取根目录 `.env` 中的配置。更细的检索器、扩展方式和缓存参数说明放在 `docs/longmemeval.md`。

LongMemEval 图检索：

```bash
./scripts/graph_lme_construct.sh
./scripts/graph_lme_run_retrieval.sh --out-dir results/graph_lme
```

HaluMem 图完整评估流程：

```bash
./scripts/graph_halu_eval_pipeline.sh
```

HaluMem flat 运行入口：

```bash
./scripts/halu_run.sh --dataset medium
```

`halu_run.sh`不包含 QA 的评估。注意：HaluMem 的评估阶段会消耗大量 judge model token，如需继续评估 flat 结果：

```bash
python -m evals.halu_eval --file_path <path-to-structure_eval_results.jsonl>
```

## 文档

更详细的说明已经迁移到 [`docs/`](docs/index.md)，并且已经按 GitHub Pages 的形式搭好了。

- [文档首页](docs/index.md)
- [Quick Start](docs/quickstart.md)
- [环境变量](docs/environment.md)
- [LongMemEval](docs/longmemeval.md)
- [HaluMem](docs/halumem.md)
- [脚本索引](docs/scripts.md)
- [GitHub Pages 部署](docs/github-pages.md)

本地预览文档：

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

## 引用

```bibtex
@article{unifiedmem2026,
  title={Does Memory Need Graphs? A Unified Framework and Empirical Analysis for Long-Term Dialog Memory},
  author={UnifiedMem Authors},
  journal={arXiv preprint arXiv:2601.01280},
  year={2026}
}
```

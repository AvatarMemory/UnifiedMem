<div align="center">

# Does Memory Need Graphs? A Unified Framework and Empirical Analysis for Long-Term Dialog Memory

<p>
  <a href="https://arxiv.org/abs/2601.01280v2"><img src="https://img.shields.io/badge/arXiv-2601.01280-B31B1B?style=for-the-badge&logo=arxiv&logoColor=red" alt="arXiv Paper"></a>
  <a href="README-zh.md"><img src="https://img.shields.io/badge/🇨🇳中文版-1a1a2e?style=for-the-badge"></a>
</p>

<p align="center">
  <img src="./README.assets/unified_framework.png" style="max-width: min(100%, 1000px); height: auto;" alt="Unified Framework Diagram">
</p>

</div>

UnifiedMem is a unified framework for long-term dialog memory research. It supports flat and graph-based pipelines across four stages:

- `index`
- `retrieve`
- `qa`
- `qa_eval`

## 🎉 News

- **2026.04.06** Our [**UnifiedMem**](https://arxiv.org/abs/2601.01280v3) is accepted by ACL 2026 Main Conference!

## Overview

This repository implements a unified framework for building and evaluating long-term dialog memory systems. It covers both flat and graph-based approaches and organizes the full workflow into indexing, retrieval, QA generation, and QA evaluation.

Key capabilities:

- unified support for flat and graph memory pipelines
- structured memory extraction, including summaries, keyphrases, and user facts
- configurable LLM backends for `index`, `qa`, and `qa_eval`
- shared embedding configuration across indexing and retrieval
- evaluation workflows for both LongMemEval and HaluMem

## 🚀 Quick Start

### 1. Install

```bash
conda create -n unifiedmem python=3.11 -y
conda activate unifiedmem
pip install -r requirements.txt
```

### 2. Create the root `.env`

```bash
cp .env.example .env
```

Minimal settings:

```dotenv
OPENAI_API_KEY=""
OPENAI_BASE_URL="http://localhost:8001/v1"
LLM_MODEL="gpt-4o-mini"
EMBEDDING_MODEL="contriever"
```

Optional stage-specific LLM overrides:

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

If a stage-specific value is empty, UnifiedMem falls back to `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `LLM_MODEL`.

### 3. Prepare datasets

Place the datasets under these directories first:

- LongMemEval: `data/longmemeval-cleaned/`
- HaluMem: `data/HaluMem/`

If you start from the original cleaned LongMemEval release, place these files first:

- `data/longmemeval-cleaned/longmemeval_s_cleaned.json`
- `data/longmemeval-cleaned/longmemeval_oracle.json`
- `data/HaluMem/HaluMem-Medium.jsonl`

For LongMemEval, run the deduplication preprocessing step before indexing or retrieval:

```bash
python data_preprocessing/lme_deduplicate.py
```

After preprocessing, the pipelines in this repository usually use:

- `data/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json`
- `data/longmemeval-cleaned/longmemeval_oracle_deduplicate.json`
- `data/HaluMem/HaluMem-Medium.jsonl`

### 4. ▶️ Run Pipeline

LongMemEval flat retrieval:

```bash
./scripts/flat_lme_build_index.sh

python -m src.flat.lme_run_retrieval \
  --out_dir results/flat_lme
```

The flat LongMemEval defaults follow the root `.env`. Detailed retriever, expansion, and cache arguments are documented in `docs/longmemeval.md`.

LongMemEval graph retrieval:

```bash
./scripts/graph_lme_construct.sh
./scripts/graph_lme_run_retrieval.sh --out-dir results/graph_lme
```

HaluMem graph full evaluation pipeline:

```bash
./scripts/graph_halu_eval_pipeline.sh
```

HaluMem flat run:

```bash
./scripts/halu_run.sh --dataset medium
```

`halu_run.sh` don't include QA-eval scoring. Note: HaluMem evaluation can consume a large number of judge-model tokens, To score flat results:

```bash
python -m evals.halu_eval --file_path <path-to-structure_eval_results.jsonl>
```

## Documentation

Detailed docs now live in [`docs/`](docs/index.md) and are ready for GitHub Pages.

- [Docs Home](docs/index.md)
- [Quick Start](docs/quickstart.md)
- [Environment Variables](docs/environment.md)
- [LongMemEval](docs/longmemeval.md)
- [HaluMem](docs/halumem.md)
- [Scripts](docs/scripts.md)
- [GitHub Pages Setup](docs/github-pages.md)

To preview the docs locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

## Citation

```bibtex
@article{unifiedmem2026,
  title={Does Memory Need Graphs? A Unified Framework and Empirical Analysis for Long-Term Dialog Memory},
  author={UnifiedMem Authors},
  journal={arXiv preprint arXiv:2601.01280},
  year={2026}
}
```

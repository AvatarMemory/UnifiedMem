# Quick Start

This page gets you from a fresh clone to a first successful run.

## 1. Install

```bash
conda create -n unifiedmem python=3.11 -y
conda activate unifiedmem
pip install -r requirements.txt
```

## 2. Create `.env`

```bash
cp .env.example .env
```

At minimum, check these values in the root `.env`:

```dotenv
OPENAI_API_KEY=""
OPENAI_BASE_URL="http://localhost:8001/v1"
LLM_MODEL="gpt-4o-mini"
EMBEDDING_MODEL="contriever"
```

If different stages need different LLM backends, add stage-specific overrides:

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

When a stage-specific value is empty, UnifiedMem falls back to `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `LLM_MODEL`.

## 3. Prepare data

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

## 4. First run: LongMemEval flat retrieval

LongMemEval flat retrieval:

```bash
./scripts/flat_lme_build_index.sh

python -m src.flat.lme_run_retrieval \
  --out_dir results/flat_lme
```

The flat LongMemEval defaults follow the root `.env`. Detailed retriever, expansion, and cache arguments are documented in [LongMemEval](longmemeval.md).

## 5. First run: LongMemEval graph retrieval

Build the graph:

```bash
./scripts/graph_lme_construct.sh
```

Run retrieval:

```bash
./scripts/graph_lme_run_retrieval.sh --out-dir results/graph_lme
```

## 6. First run: HaluMem graph full evaluation pipeline

```bash
./scripts/graph_halu_eval_pipeline.sh
```

HaluMem evaluation can consume a large number of judge-model tokens, especially in the QA and scoring stages. Start with `HaluMem-Medium`, verify your `.env` model settings first, and be cautious before launching the full evaluation pipeline.

This is the recommended one-command entry when you need both QA generation and final QA-eval scoring.

## 7. First run: HaluMem flat run

```bash
./scripts/halu_run.sh --dataset medium
```

`./scripts/halu_run.sh` runs the flat structured-memory workflow and can generate QA responses, but it does not run the separate final QA-eval scoring stage.

Useful overrides:

```bash
./scripts/halu_run.sh \
  --dataset medium \
  --embedding-model contriever \
  --llm-model qwen3-8b \
  --qa-llm qwen3-8b \
  --top-k 20
```

If you also want final QA evaluation metrics for the flat run, score the generated `structure_eval_results.jsonl` separately:

```bash
python -m evals.halu_eval --file_path <path-to-structure_eval_results.jsonl>
```

## What to read next

- [Environment Variables](environment.md)
- [LongMemEval](longmemeval.md)
- [HaluMem](halumem.md)
- [Scripts](scripts.md)

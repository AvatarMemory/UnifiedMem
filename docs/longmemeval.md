# LongMemEval

This page focuses on the current recommended LongMemEval workflows.

## Required Files

Place the LongMemEval files under `data/longmemeval-cleaned/`.

If you start from the original cleaned LongMemEval release, place these files first:

- `longmemeval_s_cleaned.json`
- `longmemeval_oracle.json`

Before indexing or retrieval, run:

```bash
python data_preprocessing/lme_deduplicate.py
```

The pipelines in this repository usually consume:

- `longmemeval_s_cleaned_deduplicate.json`
- `longmemeval_oracle_deduplicate.json`

## Recommended Workflow: Graph Retrieval

### Step 1. Construct graph

```bash
./scripts/graph_lme_construct.sh \
  --in-file data/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json \
  --embedding contriever
```

### Step 2. Run graph retrieval

```bash
./scripts/graph_lme_run_retrieval.sh \
  --in-file data/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json \
  --out-dir results/graph_lme \
  --embedding contriever \
  --graphrag-mode entity,chunk
```

### Step 3. Compute recall

```bash
python -m evals.lme_compute_recall \
  --in_file results/graph_lme/graph_retrieval_results-contriever.json \
  --oracle_file data/longmemeval-cleaned/longmemeval_oracle_deduplicate.json \
  --haystack_file data/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json
```

### Step 4. Optional QA generation

```bash
python -m evals.run_generation \
  --in_file results/graph_lme/graph_retrieval_results-contriever.json \
  --topk_context 5 \
  --out_dir results/graph_lme_generation
```

### Step 5. Optional QA evaluation

```bash
python -m evals.lme_compute_qa auto \
  results/graph_lme_generation/<your_output_file> \
  data/longmemeval-cleaned/longmemeval_oracle_deduplicate.json
```

## Flat Workflow

The flat pipeline is easiest to reason about in two stages:

1. build expansion caches
2. run flat retrieval

### Step 1. Build expansion caches

```bash
./scripts/flat_lme_build_index.sh \
  data/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json
```

### Step 2. Run flat retrieval

Recommended entry:

```bash
python -m src.flat.lme_run_retrieval \
  --in_file data/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json \
  --out_dir results/flat_lme
```

By default, the flat LongMemEval pipeline reads model settings from the root `.env`, and the default expansion cache directory is derived from the index-stage model name.

Important flags:

- `--retriever`: `flat-bm25`, `flat-contriever`, `flat-stella`, `flat-gte`, `flat-openai`
- `--index_expansion_method`: one or more expansion types
- `--index_expansion_result_join_mode`: `none`, `separate`, `merge`, or `merge_raw`
- `--index_expansion_result_cache`: cache files aligned with the expansion list
- `--value_expansion_join_mode`: whether retrieval output should include expansion text

### Optional shell shortcut

`./scripts/lme_run_retrieval.sh` is available as a convenience wrapper, but the Python module above is the clearest documented interface because it uses explicit named arguments.

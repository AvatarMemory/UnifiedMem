# HaluMem

This page documents the current HaluMem entry points.

## Required File

Expected default dataset:

- `data/HaluMem/HaluMem-Medium.jsonl`

Optional longer dataset:

- `data/HaluMem/HaluMem-Long.jsonl`

## Recommended Workflow: Graph Full Pipeline

If you want one command that covers the full graph evaluation flow, including QA generation and final QA-eval scoring, start here:

```bash
./scripts/graph_halu_eval_pipeline.sh
```

This workflow can consume a large number of judge-model tokens, especially in the QA and final scoring stages. For a first run, prefer:

- `HaluMem-Medium`
- a verified `.env` with the intended index, QA, and QA-eval models

This pipeline can perform:

1. graph construction
2. graph retrieval
3. add-memory generation
4. merged evaluation file generation
5. final scoring

Useful options:

- `--in-file`
- `--graph-root`
- `--output-dir`
- `--retrieval-out`
- `--merged-eval`
- `--score-file`
- `--tmp-dir`
- `--embedding`
- `--graphrag-mode`
- `--top-k`
- `--skip-construct`
- `--skip-retrieval`
- `--skip-add-memory`
- `--skip-gen-eval`
- `--skip-score`

## Lower-Level Graph Workflow

### Step 1. Construct graph

```bash
./scripts/graph_halu_construct.sh \
  --embedding contriever
```

### Step 2. Run retrieval

```bash
./scripts/graph_halu_run_retrieval.sh \
  --embedding contriever \
  --graphrag-mode entity,chunk,one-hot-expand \
  --top-k 20
```

### Step 3. Generate merged eval input

```bash
python -m evals.halu_graph_eval \
  --mode gen_eval \
  --graph_root <graph_root> \
  --dataset_path data/HaluMem/HaluMem-Medium.jsonl \
  --retrieve_file_path <retrieval_json> \
  --output_file <merged_eval_jsonl>
```

### Step 4. Score

```bash
python -m evals.halu_eval \
  --file_path <merged_eval_jsonl> \
  --output_file <score_json> \
  --tmp_dir <tmp_dir> \
  --max_workers 10
```

## Flat Workflow

For the flat updatable memory system:

```bash
./scripts/halu_run.sh --dataset medium
```

This entry runs the flat structured-memory workflow and can generate QA responses, but it does not invoke the separate final QA-eval scoring step in `python -m evals.halu_eval`.

Common overrides:

```bash
./scripts/halu_run.sh \
  --dataset medium \
  --embedding-model contriever \
  --llm-model qwen3-8b \
  --qa-llm gpt-4o-mini \
  --top-k 20 \
  --version exp1
```

Important options:

- `--embedding-model`
- `--retrieve-method`
- `--llm-model`
- `--qa-llm`
- `--base-url`, `--api-key`
- `--qa-api-base`, `--qa-api-key`
- `--skip-qa`
- `--resume`
- `--enable_update`

To compute final QA evaluation metrics for a flat run, score the generated `structure_eval_results.jsonl` separately:

```bash
python -m evals.halu_eval \
  --file_path <path-to-structure_eval_results.jsonl> \
  --output_file <score_json>
```

## Output Files

Typical graph pipeline outputs:

- `graph_retrieval_results-*.json`
- `add_memory_by_session.json`
- `*_eval_results.jsonl`
- `eval_stat_result.json`

Typical flat outputs include retrieval logs, extracted memory records, and `structure_eval_results.jsonl` when the run completes. Final `eval_stat_result.json` for flat runs requires a separate `python -m evals.halu_eval` step.

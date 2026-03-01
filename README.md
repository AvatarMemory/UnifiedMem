<div align="center">

# Does Memory Need Graphs? A Unified Framework and Empirical Analysis for Long-Term Dialog Memory

<div align="center">
  <div align="center">
    <p>
      <a href="https://arxiv.org/abs/2601.01280v2"><img src="https://img.shields.io/badge/Project-Homepage-00d9ff?style=for-the-badge&logo=github&logoColor=white&labelColor=1a1a2e"></a>
      <a href="https://arxiv.org/abs/2601.01280v2"><img src="https://img.shields.io/badge/arXiv-2601.01280-B31B1B?style=for-the-badge&logo=arxiv&logoColor=red" alt="arXiv Paper"></a>
      <a href="README-zh.md"><img src="https://img.shields.io/badge/🇨🇳中文版-1a1a2e?style=for-the-badge"></a>
      <a href="README.md"><img src="https://img.shields.io/badge/🇺🇸English-1a1a2e?style=for-the-badge"></a>
    </p>
  </div>
</div>

<p align="center">
  <img src="./README.assets/unified_framework.png" style="max-width: min(100%, 1000px); height: auto;" alt="Unified Framework Diagram">
</p>
<p align="center" style="margin-top:-15px; font-style:italic; color:#f3f4f6;">Unified Framework Diagram</p>
</div>

---
## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Usage](#usage)
- [System Architecture](#system-architecture)
- [Example Scripts](#example-scripts)
- [Output Format](#output-format)
- [Configuration Files](#configuration-files)
- [Development](#development)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

This repository contains the code for our paper: "Does Memory Need Graphs? A Unified Framework and Empirical Analysis for Long-Term Dialog Memory". We conduct experimental and system-oriented analysis of long-term dialog memory architectures. We propose a unified framework that deconstructs dialog memory systems into core components and supports both graph-based and non-graph (flat) approaches. Under this framework, we perform staged refinement experiments on the LongMemEval and HaluMem datasets, identifying stable and strong baselines that enable fair comparisons and practical deployment of component choices.

## Overview

This repository implements a unified framework for building and evaluating agentic, updatable memory systems for long-term dialog. Key capabilities:
- **Support for flat and graph-based indexing** - Includes mainstream non-graph (flat) and graph-based approaches for memory organization.
- **Structured Memory Extraction** — extract session-level summaries, salient keyphrases, and user facts.
- **Dynamic Memory Management** — decide when to create, update, or skip memory points using similarity search and LLM judgments.
- **Flexible Backends** — support multiple embedding, retrieval, and generation backends (Contriever, Stella, GTE, SentenceTransformers, OpenAI, BM25, gpt4o-mini, Lamma3.1-8B, OpenAI API server).
- **Evaluation Pipelines** — end-to-end tooling for LongMemEval and HaluMem: retrieval logs, recall metrics, and optional answer generation.

## Installation

Requirements (suggested)
- Python 3.10+ (3.11 recommended)
- conda or pip + virtualenv/venv
- GPU + CUDA for local large-model experiments (optional)

Quick setup — using conda (recommended)

```bash
# Create and activate a conda environment (example)
conda create -n unifiedmem python=3.11 -y
conda activate unifiedmem

# Install dependencies
pip install -r requirements.txt
```

Environment variable examples (Note: Environment variables may be overridden by settings in the configuration files)

```bash
# Set necessary environment variables
export OPENAI_API_KEY="your-api-key"  # For OpenAI models
export HF_HOME="/path/to/cache"        # Cache directory for HuggingFace models
```

## Configuration

This repository supports layered `.env` files. The loader checks (and loads) these files in order:
1. Terminal environment variables
2. `.env` in the project root directory
3. `.env` in subdirectories
4. Command line arguments

Recommended environment variables (see `.env.example`):
- `OPENAI_API_KEY`: API key for OpenAI-compatible LLM backends.
- `OPENAI_BASE_URL`: Base URL for local or third-party vLLM/OpenAI-compatible server (e.g. `http://localhost:8001/v1`).
- `LLM_MODEL`: Default model for generation (e.g. `gpt-4o-mini` or `meta-llama/Meta-Llama-3.1-8B-Instruct`).
- `EMBEDDING_MODEL` / `EMBEDDING_RETRIEVER`: Defaults for embedding selection.
- `EMBEDDING_API_URL`, `EMBEDDING_API_KEY`: Parameters for third-party embedding models.

For detailed environment variable configurations, refer to the [Configuration Files](#configuration-files) section.

## Datasets

### LongMemEval

LongMemEval (s/m) contains 500 questions, each associated with numerous dialogue sessions. Download guidance refer to the original repository. The default storage location for the dataset is `{project_root}/data/longmemeval-cleaned`.

Once downloaded, run:

```bash
python data_preprocessing/lme_deduplicate.py
```
to deduplicate session ids for each question (fix duplicate session metadata before indexing).

### HaluMem

HaluMem evaluates memory systems' ability to handle hallucinations and memory updates.

**Dataset Variants:**
- **HaluMem-Medium**: `data/HaluMem/HaluMem-Medium.jsonl` - Moderate conversation length
- **HaluMem-Long**: `data/HaluMem/HaluMem-Long.jsonl` - Extended conversations with distractors

To download HaluMem, follow the official repository [link](https://github.com/MemTensor/HaluMem). The default storage location is `{project_root}/data/HaluMem`. In our experiments, we only considered HaluMem-Medium as it is already sufficiently challenging.

---

## Usage

### LongMemEval Evaluation

#### Flat Method

##### 0. Extract Expansions
```bash
python data_preprocessing/lme_extract_summ.py
python data_preprocessing/lme_extract_keyphrase.py
python data_preprocessing/lme_extract_userfact.py
```

Note that you need to set the input data path and output expansion path in the extraction scripts or via their CLI arguments. For environment variable settings, refer to the [Configuration](#configuration) and [Configuration Files](#configuration-files) sections.

##### 1. Run Retrieval

```bash
./scripts/lme_run_retrieval.sh
```

Support no expansion and multiple expansions.

**Key Arguments:**

- `--retriever`: Choose from `flat-bm25`, `flat-contriever`, `flat-stella`, `flat-gte`, `flat-openai`
- `--granularity`: Currently supports `session` (session-level retrieval)
- `--index_expansion_method`: Comma-separated list of expansion methods:
  - `none`: No expansion (original sessions only)
  - `session-summ`: Session summaries
  - `session-keyphrase`: Session keyphrases
  - `session-userfact`: User facts from sessions
  - Combine multiple: `session-userfact,session-keyphrase,session-summ`
- `--index_expansion_result_join_mode`: How to combine expansions with original content:
  - `none`: No expansion
  - `separate`: Keep expansions separate (retrieve from each)
  - `merge`: Merge expansions with original sessions; two embeddings are computed based on the merged expansion and the original session
  - `merge_raw`: Merge raw sessions into expansions; only one embedding is computed
- `--index_expansion_result_cache`: Comma-separated cache file paths (must match expansion methods)
- `--use_raw_session_as_key`: Also use original session dialogue for retrieval
- `--value_expansion_join_mode`: Expand retrieved values with expansion content (`none` or `merge`)

**Key Differences from Original LongMemEval:**
1. Support multiple expansions
2. Support multiple expansion join methods
3. Fix the issue in LongMemEval where the empty index expansions lead to incorrectly discarding the original session
4. Fix the issue in LongMemEval where the altered session id (changing the 'answer' prefix) led to mismatched id lookups

##### 2. Compute Recall Metrics

```bash
python evals/lme_compute_recall.py \
    --in_file <retrieval_log_file> \
    --oracle_file data/longmemeval-cleaned/longmemeval_oracle_deduplicate.json \
    --haystack_file data/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json \
    --out_file <out_file>
```

##### 3. Generate Answers (Optional)

```bash
python evals/lme_run_generation.py \
    --in_file <retrieval_log_file> \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --topk_context 5 \
    --cot true \
    --out_dir <output_directory>
```

**Generation Arguments:**
- `--topk_context`: Number of top retrieved contexts to use
- `--history_format`: Format for history (`json` or `nl`)
- `--useronly`: Use only user utterances (`true` or `false`)
- `--cot`: Enable chain-of-thought reasoning (`true` or `false`)
- `--merge_key_expansion_into_value`: How to merge expansions (`none`, `merge`, `replace`)

##### 4. Evaluate QA Performance
Example:

```bash
python evals/lme_compute_qa.py gpt-4o <generation_output_file> data/longmemeval-cleaned/longmemeval_oracle_deduplicate.json
```

---

#### Graph Method

Graph retrieval consists of two steps: first construct the graph, then run the graph retrieval script. Logically, flat's `--retriever`/`--index_expansion_method` corresponds to graph's `--embedding`/`--graphrag-mode`, etc.

##### 1. Construct Graph and Run Retrieval
```bash
# Construct the graph (if not already built)
./scripts/graph_lme_construct.sh \
  --in-file data/longmemeval-cleaned/longmemeval_s_cleaned.json \
  --out-dir data/graph_s-gpt-4o-mini \
  --embedding  text-embedding-3-small \
  --entity-namespace openai_name_entities

# Run graph retrieval
./scripts/graph_lme_run_retrieval.sh \
  --in-file data/longmemeval-cleaned/longmemeval_s_cleaned.json \
  --out-dir results/graph_lme/ \
  --embedding  text-embedding-3-small \
  --graphrag-mode entity,chunk,one-hot-expand \
  --only-need-context
```

##### 2. Compute Recall Metrics
The retrieval output of graph (e.g., `graph_retrieval_results-*.json`) has a structure compatible with flat retrieval logs, so `evals/lme_compute_recall.py` can be directly reused to compute recall.
```bash
python evals/lme_compute_recall.py \
    --in_file <retrieval_log_file> \
    --oracle_file data/longmemeval-cleaned/longmemeval_oracle_deduplicate.json \
    --haystack_file data/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json \
    --out_file <out_file>
```

##### 3. Generate Answers and Evaluate QA Performance (Optional)
When using generation (QA) for evaluation, if the input is the `graph_retrieval_results-*.json` obtained from graph retrieval, it can be passed to `lme_run_generation.py`. The generation process is identical to flat.
```bash
python evals/lme_run_generation.py \
    --in_file <retrieval_log_file> \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --topk_context 5 \
    --cot true \
    --out_dir <output_directory>

python evals/lme_compute_qa.py gpt-4o <generation_output_file> data/longmemeval-cleaned/longmemeval_oracle_deduplicate.json
```

---

### HaluMem Evaluation

#### Flat Method

##### 1. Run Memory System Evaluation

```bash
./scripts/halu_run.sh
```

**Key Arguments:**

**Data Paths:**
- `--data_path`: Path to HaluMem dataset (`.jsonl` format)
- `--out_dir`: Output directory for results
- `--cache_dir`: Cache directory for models

**Memory System Configuration:**
- `--embedding_model`: Embedding model for retrieval (`contriever`, `stella`, `gte`, `all-MiniLM-L6-v2`, etc.)
- `--retrieve_method`: Memory retrieval strategy:
  - `merge`: Combine all memory components (summary, keywords, facts) into single retriever
  - `separate`: Separate retrievers for each component, then combine scores
  - `merge_raw`: Merge with raw session text
- `--llm_model`: LLM for memory operations (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct`, `gpt-4o-mini`)
- `--llm_backend`: LLM backend (`openai` for OpenAI-compatible APIs)
- `--base_url`: Base URL for LLM API (e.g., `http://localhost:8001/v1` for vLLM)
- `--temperature`: Temperature for LLM generation (default: `0.0`)

**Memory Operations:**
- `--enable_update`: Enable memory update operations (create/update/skip decisions); if False, memories are always added into the system.
- `--keep_update_note`: Keep original memory after update operation (default: True)
- `--enable_link`: Enable linking related memories by prompting an LLM
- `--use_neighbour_memories`: Use neighboring memories for to compose top-k context (must --enable_link)

**Retrieval Configuration:**
- `--top_k`: Number of memories to retrieve for QA (default: `20`)
- `--qa_retrieve_method`: QA retrieval method:
  - `flatten`: Flatten all memory points for retrieval, where individual embedding is computed for each point (default)
  - `default`: Use retrieve_method setting
- `--use_raw_session_as_key`: Include raw session session dialogue in retrieval; must be True if retrieve_method is `merge_raw`
- `--include_point_type`: Include point type (summary/keyword/fact) in retrieved context

**QA Configuration:**
- `--qa_llm`: LLM model for QA (defaults to `llm_model`)
- `--qa_api_base`: API base URL for QA LLM
- `--skip_qa`: Skip QA generation (only extract and retrieve memories)

**Processing Configuration:**
- `--version`: Version identifier for output files
- `--resume`: Resume from existing progress
- `--use_metadata_cache`: Use cached extracted metadata
- `--metadata_cache_dir`: Directory for metadata extraction cache
- `--device`: Device for embedding model (auto-detect if not specified)

**Multiprocessing:**
- `--num_workers`: Number of parallel workers for processing users (default: `1`)
- `--gpu_ids`: Comma-separated GPU IDs (e.g., `"0,1,2,3"`)

##### 2. Evaluate Results
Default configurations suggested by HaluMem is provided in  `evals/.env`/

```bash
python evals/halu_eval.py \
  --file_path <structure_eval_results.jsonl>
```

---

#### Graph Method

##### 1. Construct and Run Graph Retrieval
```bash
# Construct and run graph retrieval
./scripts/graph_halu_construct.sh --in-file data/HaluMem/HaluMem-Medium.jsonl --llm-model gpt-4o-mini
./scripts/graph_halu_run_retrieval.sh \
 --graph-root data/nc-graph_halu_mem_medium-4o-mini \
 --out-dir results/graph_halu/ \
 --embedding text-embedding-3-small \
 --graphrag-mode entity,chunk,one-hot-expand \
 --only-need-context
```

##### 2. Generate Intermediate Files and Run Offline Evaluation (Recommended Workflow)
The original evaluation code for HaluMem used an online approach, building the graph while recording experimental results. For modularity considerations, we recommend using an offline approach for this part of the evaluation. Offline evaluation for the graph method requires first generating intermediate files (e.g., `add_memory_by_session.json`) using the `--mode` parameter of `evals/halu_graph_eval.py`. Then, merge the retrieval results with these intermediate files to form the offline evaluation input, and finally run the evaluation script to compute metrics. Recommended steps:

1. Generate add_memory (parse GraphML, output `add_memory_by_session.json`)
```bash
python evals/halu_graph_eval.py --mode add_memory \
  --graph_root data/nc-graph_halu_mem_medium-4o-mini \
  --out_path <output_file_directory>
```
This command writes `add_memory_by_session.json` to the location specified by `--out_path` (if `--out_path` is not provided, the script will default to writing under `--graph_root`).

2. Specify the retrieval result file (filename or full path) with `--retrieve_file_path`, or place the graph retrieval output (`graph_retrieval_results-*.json`) in the same directory (`data/nc-graph_halu_mem_medium-4o-mini/`). Then run the merge to generate the offline evaluation input (`*_test_eval_results.jsonl`). It is recommended to explicitly specify the output file `--out_path` in the command line:
```bash
python evals/halu_graph_eval.py --mode gen_eval \
  --graph_root data/nc-graph_halu_mem_medium-4o-mini \
  --retrieve_file_path <retrieval_file_path> \
  --out_path <output_file_directory> \
  --dataset_path <HaluMem_dataset_path> \
  [--use_entity]
```
- `--retrieve_file_path`: Path or filename of the retrieval result file (relative filenames will be searched under `--graph_root`)
- `--out_path`: Output directory. The script will write the result file inferred from the retrieval filename under this directory, e.g., `graph_retrieval_results-xxx.json` → `graph_retrieval_results-xxx_test_eval_results.jsonl`. If a path ending with `.json` or `.jsonl` is provided, it is treated as a full file path (backward compatibility).
- `--out_suffix`: Suffix for the merged output file (default `_test_eval_results.jsonl`, used for inference only when a specific output filename is not provided)
- `--dataset_path`: Path to the HaluMem dataset
- `--use_entity`: Optional, enable entity-based context construction (otherwise use chunk-based context)

This step writes a merged JSONL file (one line per user's evaluation input) in the directory specified by `--out_path` (or the default location next to the retrieval file), to be used for downstream offline evaluation.

3. Run Offline Evaluation
Provide the JSONL generated in the previous step to `evals/halu_eval.py` for metric calculation. For example:
```bash
python evals/halu_eval.py --file_path <path/to/*_eval_results.jsonl>
```
Depending on the actual file organization, you can also move the generated `*_test_eval_results.jsonl` to a separate `results` directory and point `--file_path` to its location.

Notes:
- `--mode` supports `add_memory` (parse GraphML), `gen_eval` (merge and generate evaluation JSONL), `test_llm` (quick test LLM calls).
- Ensure that the `--out_dir` from `graph_halu_run_retrieval.sh` or the location where you move/copy the retrieval results matches the path expected by `halu_graph_eval.py` (`data/nc-graph_halu_mem_medium-4o-mini/`). Alternatively, adjust the script parameters/environment variable `PROJECT_ROOT` accordingly to correctly read the retrieval results and generate offline evaluation files.

---

### General Evaluation Notes

The evaluation scripts under the `evals/` directory in this repository are common to both flat and graph retrieval/recall evaluations. The main differences lie in how the retrieval output is generated and the additional graph files (GraphML).

**General Workflow (Applicable to LongMemEval and HaluMem):**
1.  **Run Retrieval** (flat or graph) to generate retrieval logs (flat typically outputs retrieval logs in JSON/JSONL, graph outputs `graph_retrieval_results-*.json` and additionally produces GraphML files as graph construction results).
2.  **Compute Recall**: Use scripts like `evals/lme_compute_recall.py`.
3.  **Generate Answers and Evaluate QA** (Optional): Use the same generation scripts as flat (`lme_run_generation.py`, `evals/lme_compute_qa.py` or `halu_eval.py`).

**Key Points:**
- **Commonality**: Both flat and graph produce retrieval logs that can be processed by the generic scripts under `evals/` (the recall/QA pipeline can be reused).
- **Differences**: The graph pipeline additionally outputs GraphML files as graph construction results. For HaluMem evaluation, an offline approach is used. The graph retrieval generation script names and output paths are typically different (scripts with the `graph_*` prefix).
- **Recommendation**: When comparing flat and graph, keep the same oracle/haystack input files and top_k settings to ensure comparability of evaluation results.

## System Architecture
### Graph Core Components

1. **GraphRAG** (`src/graph/graphrag.py`): Core implementation of graph retrieval, responsible for graph storage selection, graph construction flow, clustering, and query parameter management.

2. **Graph Construction Entry Points** (`src/graph/lme_construct_graph.py`, `src/graph/halu_construct_graph.py`): Convert LongMemEval / HaluMem sessions or questions into graph representations and write to the working directory (including GraphML files and graph storage).

3. **Graph Retrieval Entry Points** (`src/graph/lme_run_retrieval.py`, `src/graph/halu_run_retrieval.py`): Use `GraphRAG` to perform graph-level queries, aggregate retrieval results on the graph, and output JSON retrieval logs.

4. **Entity and Graph Operations** (`src/graph/entity_extraction/extract.py`, `src/graph/_op.py`): Concrete implementations for extracting entities/relations from session text, merging nodes/edges, and linking to the knowledge graph.

5. **LLM and Embedding Adapters** (`src/graph/_llm.py`, `src/graph/_utils.py`): Embedding dispatch, adapter logic for calling OpenAI / local LLMs, and graph-related utility functions.

6. **Graph Storage Base** (`src/graph/base.py`): Abstract graph storage interface and basic implementations (e.g., NetworkX storage adapter, etc.) for reading/writing nodes/edges, querying, and clustering support.

Note: The GraphML files produced by the graph components can be used for offline inspection and visualization. `evals/halu_graph_eval.py` provides an evaluation flow based on GraphML.

### Flat Core Components

1. **AgenticMemorySystem** (`agentic_memory_system.py`): Core memory management system
   - Structured memory notes with summary, keywords, and facts
   - Dynamic memory operations (create/update/merge)
   - Multiple embedding model support
   - Memory linking and neighbor-aware operations

2. **LLMController** (`llm_controller.py`): LLM interface abstraction
   - Unified interface for different LLM backends
   - Structured output parsing with retry logic
   - Token counting and context management

3. **StructuredMemoryRetriever** (`halu_utils.py`): HaluMem-specific wrapper
   - Session-based memory management
   - Metadata extraction and caching
   - Memory operation tracking

4. **FlexibleEmbeddingRetriever** (`agentic_memory_system.py`): Retrieval backend
   - Multi-model support (Contriever, Stella, GTE, SentenceTransformer, OpenAI, BM25)
   - Efficient batch encoding
   - Similarity-based retrieval

### Retrieval Methods

- **`merge`**: Combine all memory components into a single retrieval index
- **`separate`**: Maintain separate indices for summary, keywords, and facts, then aggregate scores
- **`merge_raw`**: Include raw session text along with structured components

### Memory Update Strategies

When a new session is processed, the system:
1. Extracts structured metadata (summary, keywords, facts)
2. Retrieves similar existing memories
3. LLM decides: **create** new memory, **update** existing, or **skip** (redundant)
4. Updates are merged while preserving all original session IDs for evaluation

## Example Scripts

### LongMemEval with Multiple Expansions

```bash
bash scripts/lme_run_retrieval.sh \
    data/longmemeval-cleaned/longmemeval_m_cleaned.json \
    flat-stella \
    session \
    session-userfact,session-keyphrase,session-summ \
    merge \
    "data/longmemeval-cleaned/expansions-llama3.1_8b/session-userfact.json,data/longmemeval-cleaned/expansions-llama3.1_8b/session-keyphrase.json,data/longmemeval-cleaned/expansions-llama3.1_8b/session-summ.json" \
    llama-3.1-8b-instruct-ICL \
    my_experiment
```

### HaluMem with Full Pipeline

```bash
# Run evaluation
bash scripts/halu_run.sh \
    --dataset long \
    --embedding-model stella \
    --retrieve-method merge \
    --llm-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --base-url http://localhost:8001/v1 \
    --enable-update \
    --keep-update-note \
    --use-metadata-cache \
    --resume \
    --version full_pipeline_exp
```

### Graph Usage Examples

Below shows how to use the scripts provided in the repository to construct graphs and run graph-based retrieval (for LongMemEval / HaluMem). The output includes GraphML files per question/session and `graph_retrieval_results-*.json`.

```bash
# Construct graph for LongMemEval
./scripts/graph_lme_construct.sh \
  --in-file data/longmemeval-cleaned/longmemeval_s_cleaned.json \
  --out-dir data/graph_s-gpt-4o-mini \
  --embedding  text-embedding-3-small \
  --entity-namespace openai_name_entities

# Run graph-based retrieval for LongMemEval
./scripts/graph_lme_run_retrieval.sh \
  --in-file data/longmemeval-cleaned/longmemeval_s_cleaned.json \
  --out-dir results/graph_lme/ \
  --embedding  text-embedding-3-small \
  --graphrag-mode entity,chunk,one-hot-expand \
  --only-need-context

# Construct graph and run retrieval for HaluMem
./scripts/graph_halu_construct.sh \
 --in-file data/HaluMem/HaluMem-Medium.jsonl \
 --llm-model gpt-4o-mini

./scripts/graph_halu_run_retrieval.sh \
 --graph-root data/nc-graph_halu_mem_medium-4o-mini \
 --out-dir results/graph_halu/ \
 --embedding text-embedding-3-small \
 --graphrag-mode entity,chunk,one-hot-expand \
 --only-need-context
```

Notes:
- Use the `--graphrag-mode` flag (or `GRAPHRAG_MODE` environment variable) to select graph components (examples: `entity`, `chunk`, `one-hot-expand`, `rank-entity`).
- The output directory contains GraphML files for offline inspection and `graph_retrieval_results-*.json` retrieval logs for use by `evals/halu_graph_eval.py`.

## Output Format

### LongMemEval Retrieval Logs

Each line in the output file contains:
```json
{
  "question_id": "user123_session5_q1",
  "question_type": "single_event",
  "question": "What did I order for lunch?",
  "answer": "You ordered a chicken salad.",
  "question_date": "2024-03-15",
  "haystack_dates": ["2024-03-10", "2024-03-12", ...],
  "haystack_session_ids": ["user123_session1", "user123_session2", ...],
  "answer_session_ids": ["user123_session3"],
  "retrieval_results": {
    "query": "What did I order for lunch?",
    "ranked_items": [
      {
        "corpus_id": "user123_session3",
        "text": "...",
        "timestamp": "2024-03-12",
        "is_original": true,
        "expansion_type": "original"
      },
      ...
    ]
  }
}
```

### HaluMem Results

Results are saved in `<out_file_path>/`:
- `*_eval_results.jsonl`: Per-question results with retrieved contexts and generated answers
- `user_*.json` (in `tmp/`): Intermediate user-level results with memory operations
- Memory operation logs and statistics

## Configuration Files

This repository drives runtime configuration through environment variables and layered `.env` files. Configuration loading order:
1. Terminal environment variables
2. `.env` in the project root directory
3. `.env` in subdirectories (e.g., `evals/`)
4. Command line arguments

Below are the commonly used environment variables in the repository (defaults can be found in `.env.example` in the repository root):

- `OPENAI_API_KEY`: API Key for OpenAI or OpenAI-compatible backends (default: empty string).
  - Purpose: For calling OpenAI API, third-party compatible servers, or proxies (e.g., vLLM/OpenAI-compat services).

- `OPENAI_BASE_URL`: Base URL for OpenAI-compatible services (default: `http://localhost:8001/v1`).
  - Purpose: Set when using self-hosted vLLM/OpenAI-compat services, e.g., `http://localhost:8001/v1`.

- `LLM_MODEL`: Default model for generation (default: `gpt-4o-mini`).
  - Examples: `gpt-4o-mini`, `gpt-4o`, `meta-llama/Meta-Llama-3.1-8B-Instruct`.

- `EMBEDDING_MODEL`: Embedding model for vectorization (default: `text-embedding-3-small`).
- `EMBEDDING_RETRIEVER`: Embedding retriever selection (default: `flat-openai`).

- `EMBEDDING_API_URL`: Optional third-party embedding service URL (default: empty).
- `EMBEDDING_API_KEY`: API Key for `EMBEDDING_API_URL` (default: empty).
  - Note: When using an external embedding provider, the system POSTs `{ "model": ..., "input": [...] }` and expects an OpenAI-like response structure.

- `CACHE_DIR`: Model/data cache directory (default: `data/cache`).

- `NUM_WORKERS`: Default number of processes for multiprocessing/parallel tasks (default: `64`, can be adjusted for data preprocessing scenarios).
- `SAVE_EVERY`: Checkpoint saving frequency for long tasks (default: `256`).

- `LLM_TEMPERATURE`: Default temperature for LLM inference (default: `0.0`).
- `QA_LLM`: Default LLM for QA (default: `gpt-4o-mini`).

- `KEYPHRASE_MAX_TOKENS`, `KEYPHRASE_TEMPERATURE`: Default parameters for keyphrase extraction (defaults: `100`, `0.0` respectively).
- `SUMMARY_MAX_TOKENS`, `SUMMARY_TEMPERATURE`: Default parameters for summary extraction (defaults: `500`, `0.0` respectively).
- `USERFACT_MAX_TOKENS`, `USERFACT_TEMPERATURE`: Default parameters for user fact extraction (defaults: `2000`, `1.0` respectively).

Example: Setting environment variables in Linux/macOS shell:
```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="http://localhost:8001/v1"
```

Recommendation: Place project-level defaults in the root directory `.env` file (the repository already includes `.env.example`), and place overriding `.env` files specific to sub-processes (e.g., `evals/`, `data_preprocessing/`) in the respective subdirectories.

Below explains the current repository's support for models:
### Model Support

**Embedding Models:**
- `contriever`: Facebook Contriever
- `stella`: Stella-1.5B-v5
- `gte`: GTE-Qwen2-7B-instruct
- `all-MiniLM-L6-v2`, `all-mpnet-base-v2`: SentenceTransformers
- `openai`: OpenAI embeddings (default to text-embedding-3-small/small)
- `bm25`: BM25 sparse retrieval

**LLM Models:**
- OpenAI models: `gpt-4o-mini`, `gpt-4`, etc.
- Local models via vLLM: `meta-llama/Meta-Llama-3.1-8B-Instruct`, etc.
- Third-party API services configured via environment variables

## Development

### Project Structure

```
.
├── README.md                   # documentation
├── data/                       # Raw data and processed outputs (HaluMem, LongMemEval, etc.)
├── data_preprocessing/         # Data preprocessing and expansion scripts
│   └── lme_deduplicate.py      # LongMemEval deduplication example
├── src/                        # Core code: flat and graph pipelines
│   ├── config.py               # Environment and configuration loader (layered .env)
│   ├── flat/                   # Non-graph (embedding-based) implementation
│   │   ├── agentic_memory_system.py
│   │   ├── halu_run.py
│   │   ├── halu_utils.py
│   │   └── llm_controller.py
│   └── graph/                  # Graph-based pipeline (GraphRAG, etc.)
│       ├── graphrag.py
│       ├── lme_construct_graph.py
│       ├── halu_construct_graph.py
│       └── lme_run_retrieval.py
├── evals/                      # Evaluation scripts (recall, QA, graph eval)
│   ├── halu_eval.py
│   ├── halu_graph_eval.py
│   ├── lme_compute_recall.py
│   └── lme_compute_qa.py
├── scripts/                    # Example run scripts (construct/retrieve/evaluate)
│   ├── halu_run.sh
│   ├── lme_run_retrieval.sh
│   ├── graph_lme_construct.sh
│   └── graph_lme_run_retrieval.sh
├── model_cache/                # Local model cache (hub)
├── sample_data/                # Small sample data
└── README.assets/              # Documentation images/resources
```

Note: The above is a simplified view of the repository. The actual directory may contain additional scripts, configuration files, and output directories (e.g., `data/*`, model snapshots under `checkpoints/`, etc.).

### Running with vLLM

To use local LLMs via vLLM:

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8001 \
    --tensor-parallel-size 1

# Run evaluation pointing to vLLM server
python halu_run.py \
    --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --base_url http://localhost:8001/v1 \
    ...
```

## Citation

<!-- TODO: Add citation information once paper is published -->



## Acknowledgments

This work builds upon:
- **LongMemEval**: Long-term memory evaluation benchmark [link](https://github.com/xiaowu0162/LongMemEval)
- **HaluMem**: Hallucination-aware memory evaluation dataset [link](https://github.com/MemTensor/HaluMem)
- **A-Mem**: Agentic updatable memory system [link](https://github.com/agiresearch/A-mem)
- **nano-graph**: Simplified GraphRAG implementation [link](https://github.com/gusye1234/nano-graphrag)

## Contact

<!-- TODO: Add contact information -->
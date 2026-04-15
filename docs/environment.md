# Environment Variables

UnifiedMem is documented around a single root `.env` file:

- use `.env.example` as the template
- place `.env` in the repository root
- treat command-line arguments as explicit overrides

## Minimal `.env`

```dotenv
OPENAI_API_KEY=""
OPENAI_BASE_URL="http://localhost:8001/v1"
LLM_MODEL="gpt-4o-mini"
EMBEDDING_MODEL="contriever"
```

## Resolution Rules

For LLM-related stages, the repository uses this fallback pattern:

1. stage-specific variable
2. global variable
3. code default

Example for QA:

1. `QA_API_KEY`
2. `OPENAI_API_KEY`
3. script or code default

Empty strings are treated as unset, so they still fall back.

## Global LLM Defaults

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Global API key for OpenAI-compatible LLM services |
| `OPENAI_BASE_URL` | Global API base URL |
| `LLM_MODEL` | Global default LLM model |

## Stage-Specific LLM Overrides

### `index`

| Variable | Purpose |
| --- | --- |
| `INDEX_API_KEY` | API key for indexing-time LLM calls |
| `INDEX_BASE_URL` | Base URL for indexing-time LLM calls |
| `INDEX_LLM_MODEL` | LLM model used in indexing-related steps |

### `qa`

| Variable | Purpose |
| --- | --- |
| `QA_API_KEY` | API key for QA generation |
| `QA_BASE_URL` | Base URL for QA generation |
| `QA_LLM_MODEL` | LLM model used for QA generation |

### `qa_eval`

| Variable | Purpose |
| --- | --- |
| `QA_EVAL_API_KEY` | API key for QA evaluation |
| `QA_EVAL_BASE_URL` | Base URL for QA evaluation |
| `QA_EVAL_LLM_MODEL` | Judge model used in QA evaluation |

## Embedding Configuration

| Variable | Purpose |
| --- | --- |
| `EMBEDDING_MODEL` | Shared embedding backend or model name |
| `EMBEDDING_RETRIEVER` | Default flat retriever backend |
| `EMBEDDING_API_URL` | External embedding API endpoint |
| `EMBEDDING_API_KEY` | API key for the external embedding API |

`retrieve` usually shares the same embedding configuration as `index`, so there is no separate retrieve-stage embedding block in `.env.example`.

## Common Runtime Settings

| Variable | Purpose |
| --- | --- |
| `CACHE_DIR` | Model and intermediate cache directory |
| `NUM_WORKERS` | Worker count for preprocessing and some evaluation jobs |
| `SAVE_EVERY` | Save interval for long preprocessing runs |
| `TOP_K` | Retrieval top-k for scripts that read it from env |
| `LLM_TEMPERATURE` | Default LLM temperature where applicable |

## Dataset and Path Settings

| Variable | Purpose |
| --- | --- |
| `DATA_DIR` | Root data directory |
| `GRAPH_ROOT` | Generic graph output directory override |
| `HALU_DATA_PATH` | Explicit HaluMem input file |
| `HALU_GRAPH_ROOT` | Explicit HaluMem graph root |
| `GRAPHRAG_MODE` | Default graph retrieval mode list |

## Recommended Pattern

Use one of these two styles:

### Style A: one backend for everything

```dotenv
OPENAI_API_KEY="..."
OPENAI_BASE_URL="http://localhost:8001/v1"
LLM_MODEL="qwen3-8b"
EMBEDDING_MODEL="contriever"
```

### Style B: separate backends by stage

```dotenv
OPENAI_API_KEY="..."
OPENAI_BASE_URL="http://localhost:8001/v1"
LLM_MODEL="qwen3-8b"

INDEX_LLM_MODEL="qwen3-8b"
QA_LLM_MODEL="gpt-4o-mini"
QA_EVAL_LLM_MODEL="gpt-4o-mini"

EMBEDDING_MODEL="contriever"
```

Keep the root `.env` short. If a value is only used once, prefer passing it on the command line instead of permanently adding more environment variables.

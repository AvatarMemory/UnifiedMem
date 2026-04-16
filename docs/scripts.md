# Scripts

This page maps each recommended script to its role.

## Recommended Entry Scripts

| Script | Use it for | Notes |
| --- | --- | --- |
| `scripts/flat_lme_build_index.sh` | Build LongMemEval flat expansion caches | Generates summary, keyphrase, and userfact caches |
| `scripts/graph_lme_construct.sh` | Build LongMemEval graph artifacts | Recommended graph indexing entry |
| `scripts/graph_lme_run_retrieval.sh` | Run LongMemEval graph retrieval | Clean named-argument interface |
| `scripts/halu_run.sh` | Run HaluMem flat memory evaluation | Main flat HaluMem entry; final QA-eval scoring is a separate step |
| `scripts/graph_halu_construct.sh` | Build HaluMem graph artifacts | Graph indexing for HaluMem |
| `scripts/graph_halu_run_retrieval.sh` | Run HaluMem graph retrieval | Produces retrieval JSON |
| `scripts/graph_halu_eval_pipeline.sh` | Run end-to-end HaluMem graph evaluation | Best single-command graph pipeline |

## LongMemEval

### Flat

- `scripts/flat_lme_build_index.sh`
- `scripts/lme_run_retrieval.sh`

Recommended pattern:

1. use `flat_lme_build_index.sh` for caches
2. use `python -m src.flat.lme_run_retrieval` for explicit retrieval arguments

### Graph

- `scripts/graph_lme_construct.sh`
- `scripts/graph_lme_run_retrieval.sh`

## HaluMem

### Flat

- `scripts/halu_run.sh`

### Graph

- `scripts/graph_halu_construct.sh`
- `scripts/graph_halu_run_retrieval.sh`
- `scripts/graph_halu_eval_pipeline.sh`

## Lower-Level Python Entry Points

Use these when you want named arguments or to compose your own workflow:

| Module | Purpose |
| --- | --- |
| `python -m src.flat.lme_run_retrieval` | Flat LongMemEval retrieval |
| `python -m src.graph.lme_construct_graph` | LongMemEval graph construction |
| `python -m src.graph.lme_run_retrieval` | LongMemEval graph retrieval |
| `python -m src.flat.halu_run` | Flat HaluMem evaluation |
| `python -m src.graph.halu_construct_graph` | HaluMem graph construction |
| `python -m src.graph.halu_run_retrieval` | HaluMem graph retrieval |
| `python -m evals.run_generation` | Generation for retrieval outputs |
| `python -m evals.lme_compute_qa` | QA evaluation for LongMemEval outputs |
| `python -m evals.halu_graph_eval` | HaluMem graph intermediate generation |
| `python -m evals.halu_eval` | HaluMem final scoring |

## Documentation Rule for This Repo

When in doubt:

- prefer scripts for the common path
- prefer Python modules for explicit named parameters
- prefer `docs/` over expanding the root README again

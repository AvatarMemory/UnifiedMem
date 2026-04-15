# UnifiedMem Documentation

UnifiedMem is a unified framework for long-term dialog memory research. The repository supports both flat and graph-based pipelines for indexing, retrieval, QA generation, and QA evaluation on LongMemEval and HaluMem.

## Read This First

- Start with [Quick Start](quickstart.md) if you want a working run as quickly as possible.
- Read [Environment Variables](environment.md) before changing models, endpoints, or API keys.
- Use [LongMemEval](longmemeval.md) and [HaluMem](halumem.md) for task-specific workflows.
- Check [Scripts](scripts.md) when you want to know which entry point is recommended.
- Follow [GitHub Pages](github-pages.md) to publish this `docs/` folder as an online site.

## Recommended Workflow

1. Install dependencies.
2. Copy `.env.example` to `.env` in the repository root.
3. Fill in your model, endpoint, and API settings.
4. Prepare datasets under `data/`.
5. Run one of the recommended pipelines:
   - LongMemEval graph retrieval
   - HaluMem graph full pipeline
   - HaluMem flat memory evaluation

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/` | Core implementation for flat and graph memory systems |
| `scripts/` | Recommended shell entry points |
| `evals/` | QA and memory evaluation scripts |
| `data_preprocessing/` | Data cleanup and expansion-building utilities |
| `docs/` | MkDocs source for GitHub Pages |

## Documentation Philosophy

This documentation only describes the current recommended usage:

- one root `.env`
- current script parameters
- current recommended entry points

It is intentionally shorter and stricter than the old README so the repo is easier to navigate.

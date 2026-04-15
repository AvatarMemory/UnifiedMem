# GitHub Pages

This repository now includes a GitHub Pages-ready MkDocs setup.

## Files Added for Docs Publishing

| Path | Purpose |
| --- | --- |
| `mkdocs.yml` | MkDocs site configuration |
| `docs/` | Markdown source for the site |
| `docs/requirements.txt` | Lightweight docs-only dependencies |
| `.github/workflows/docs.yml` | GitHub Actions workflow for Pages deployment |

## Preview Locally

Install docs dependencies:

```bash
pip install -r docs/requirements.txt
```

Serve locally:

```bash
mkdocs serve
```

Build a static site:

```bash
mkdocs build --strict
```

## Enable GitHub Pages

In the repository settings:

1. Open `Settings`.
2. Open `Pages`.
3. Set the source to `GitHub Actions`.

After that, pushing to `main` or `master` will trigger `.github/workflows/docs.yml`.

## Before the First Public Deployment

Update these fields in `mkdocs.yml`:

- `site_url`
- `repo_url`
- `repo_name` if needed

Replace the placeholder GitHub username with the real repository owner.

## Suggested Documentation Growth

Good future pages to add:

- experiment recipes
- model/backend combinations
- output file schemas
- troubleshooting
- reproducibility notes

Keep the root README short and move detailed operational notes into `docs/`.

# odyssey

A small project exploring Homer’s Odyssey text, RAG-style tooling, and an early Gradio comparison app.

## Motivation

- I liked reading The Odyssey (Chapman’s was not the easiest)
- I’m excited for Nolan’s adaptation
- I wanted an excuse to experiment with RAG + LLM + graph workflows

## What’s here

- `chat.py` — a simple Gradio app for comparing two model outputs side-by-side.
- `scripts/1_fetch_odyssey.py` — fetches the Odyssey source text and splits it into reference and material files.
- `scripts/2_split_books.py` — splits `data/odyssey.txt` into individual book files under `data/books/`.
- `src/odyssey/` — local Python package code, currently exposing `OllamaEmbeddings` for Chromadb integration.
- `data/` — text source files and generated book splits.
- `db/` — local Chroma database storage.
- `notebooks/dev.ipynb` — exploratory notebook

## Requirements

- Python >= 3.13
- Project dependencies are declared in `pyproject.toml`.

## Quick start

1. Create and activate your Python environment.

```bash
uv venv
```

2. Install dependencies:

```bash
uv sync
```

3. Fetch the Odyssey text:

```bash
uv run scripts/1_fetch_odyssey.py
```

4. Split the downloaded text into books:

```bash
uv run scripts/2_split_books.py

```

5. Run the Gradio app:

```bash
uv run chat.py
```

## Notes

- `scripts/1_fetch_odyssey.py` expects `data/odyssey.mb.txt` and writes `data/odyssey.txt` plus `data/reference.txt`.
- `scripts/2_split_books.py` requires `data/odyssey.txt` and outputs one file per book to `data/books/`.

## Next steps

- Add a real LLM backend to `chat.py`.
- Wire `src/odyssey` package into the app or notebook workflows.
- Refine README with installation, data, and architecture details as the project evolves.

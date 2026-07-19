# PDF Chatbot

A command-line chatbot that answers questions about one or more PDFs using local models via [Ollama](https://ollama.com) — no API keys, no data leaving your machine. Pick PDFs, ask questions (including follow-ups), get answers grounded strictly in those documents (not the model's general knowledge), with page citations.

## How it works

1. Pick one or more PDFs via a file dialog.
2. Each PDF is split into chunks and tagged with the section/heading it came from (e.g. `II. GENERAL EMPLOYMENT INFORMATION > B. Vacation Benefits > 2. Accrual`), so retrieval can tell similar-sounding subsections apart — even across different PDFs.
3. Chunks are embedded and cached in a per-PDF FAISS index — the same PDF won't be re-embedded on a later run — then merged into one combined index for the session if multiple PDFs were selected.
4. Each question is answered by: resolving follow-ups ("what about X?") against recent chat history into a standalone question, retrieving candidate chunks, capping how many can come from the same section (so one section — in one document — can't crowd out others), having the LLM rerank the survivors, then generating an answer strictly from that retrieved context — with an explicit instruction to say "not in this document" rather than guess. The chat history only steers retrieval; the answer itself never comes from history, just the documents.

For the full technical breakdown (constants, function-by-function pipeline, known gotchas), see [CLAUDE.md](CLAUDE.md).

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- Two Ollama models pulled:
  ```
  ollama pull llama3
  ollama pull nomic-embed-text
  ```
  (`llama3` handles chat/reranking; `nomic-embed-text` handles embeddings — `llama3` alone can't do embeddings in Ollama.)

## Setup

```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

## Usage

```
python main.py
```

On Windows, you can instead run `.\run.ps1`, which launches `main.py` using the project venv's Python directly — handy since PowerShell doesn't keep a venv activated across new terminal windows.

A file picker opens (you can select more than one PDF — hold Ctrl/Shift while selecting) — then ask questions at the `You:` prompt, including follow-ups like "what about X?". Type `exit` to quit.

## Project structure

```
main.py                              the working chatbot
requirements.txt                     pinned dependencies
run.ps1                              Windows launcher (uses venv Python directly)
CLAUDE.md                            detailed architecture notes for AI coding assistants
experiments/hybrid_retrieval_demo.py a prototype combining FAISS + BM25 retrieval — intentionally
                                      unfinished, not wired into main.py (see its docstring)
faiss_indexes/                       generated per-PDF vector index cache (created on first run)
```

## Known limitations

- No `pyproject.toml`/packaging yet — this is a script, not an installable package.
- The local LLM (llama3, 8B) can sometimes misread facts out of PDF tables that got flattened into plain text during extraction — e.g. failing to connect a tenure tier like "Year 11 and thereafter" to its own figure on the next line. This is a model-capability limit, not a bug in the retrieval pipeline.
- Chat history is per-session only — it isn't saved to disk, so it's gone once you exit.

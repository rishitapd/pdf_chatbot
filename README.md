# PDF Chatbot

A command-line chatbot that answers questions about a PDF using local models via [Ollama](https://ollama.com) — no API keys, no data leaving your machine. Pick a PDF, ask questions, get answers grounded strictly in that document (not the model's general knowledge), with page citations.

## How it works

1. Pick a PDF via a file dialog.
2. The PDF is split into chunks and tagged with the section/heading it came from (e.g. `II. GENERAL EMPLOYMENT INFORMATION > B. Vacation Benefits > 2. Accrual`), so retrieval can tell similar-sounding subsections apart.
3. Chunks are embedded and cached in a per-PDF FAISS index — the same PDF won't be re-embedded on a later run.
4. Each question is answered by: retrieving candidate chunks, capping how many can come from the same section (so one section can't crowd out others), having the LLM rerank the survivors, then generating an answer strictly from that context — with an explicit instruction to say "not in this document" rather than guess.

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
pip install langchain langchain-community langchain-ollama langchain-core faiss-cpu
```

There's no `requirements.txt` yet — install the packages above manually.

## Usage

```
python main.py
```

On Windows, you can instead run `.\run.ps1`, which launches `main.py` using the project venv's Python directly — handy since PowerShell doesn't keep a venv activated across new terminal windows.

A file picker opens — select a PDF, then ask questions at the `You:` prompt. Type `exit` to quit.

## Project structure

```
main.py                              the working chatbot
run.ps1                              Windows launcher (uses venv Python directly)
CLAUDE.md                            detailed architecture notes for AI coding assistants
experiments/hybrid_retrieval_demo.py a prototype combining FAISS + BM25 retrieval — intentionally
                                      unfinished, not wired into main.py (see its docstring)
faiss_indexes/                       generated per-PDF vector index cache (created on first run)
```

## Known limitations

- No `requirements.txt`/`pyproject.toml` yet.
- The local LLM (llama3, 8B) can sometimes misread facts out of PDF tables that got flattened into plain text during extraction — e.g. failing to connect a tenure tier like "Year 11 and thereafter" to its own figure on the next line. This is a model-capability limit, not a bug in the retrieval pipeline.
- Single PDF per session; no conversation memory across questions yet.

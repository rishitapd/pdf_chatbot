# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A command-line PDF chatbot. It loads a PDF (via a file picker dialog), splits it into chunks, embeds the chunks with a local Ollama model, stores/loads the embeddings in a per-PDF FAISS index, and answers user questions in a REPL loop using retrieval-augmented generation — answers are restricted to the retrieved PDF context, not the LLM's general knowledge. Follow-up questions ("what about X?") are resolved against recent chat history before retrieval, but the final answer is still grounded strictly in retrieved document text, not the chat history itself.

[main.py](main.py) is the working chatbot; there is no package structure, test suite, or build tooling. [experiments/hybrid_retrieval_demo.py](experiments/hybrid_retrieval_demo.py) is a separate, intentionally-not-working prototype exploring hybrid (FAISS + BM25) retrieval — see its module docstring for what's unfinished there. Don't port fixes from `main.py` into it or vice versa without checking both are still meant to diverge.

## Running

```
python main.py
```
or, on Windows, `.\run.ps1` — launches `main.py` with the project's venv Python directly, without needing to activate the venv first (useful since PowerShell doesn't persist activation across new terminal windows).

This launches a Tkinter file-picker dialog — select a PDF file to begin. There are no CLI arguments; the PDF is chosen interactively.

Requires a local [Ollama](https://ollama.com) server running with both:
- `llama3` pulled (`ollama pull llama3`) — used for chat/generation and reranking (`LLM_MODEL` in main.py)
- `nomic-embed-text` pulled (`ollama pull nomic-embed-text`) — used for embeddings (`EMBEDDING_MODEL` in main.py). `llama3` cannot be used for embeddings: it only declares Ollama's `completion` capability, not `embedding`, and will fail with a 501 error if you try.

There is now a `requirements.txt` — `pip install -r requirements.txt` covers `main.py`'s dependencies plus `rank-bm25` (only needed by the experiments demo).

There is no lint, test, or build command configured for this project.

## Architecture

Everything runs inside `main()` (guarded by `if __name__ == "__main__":`), executed once per process:

1. **PDF selection & load** — `load_user_pdf()` opens a native file dialog and loads pages with `PyPDFLoader`.
2. **Section tagging** — `tag_pages_with_sections()` does best-effort heading detection (numbered/lettered patterns like `I.`, `A.`, `1.`) across all pages before chunking, recording an offset-ordered list of headings per page rather than a single page-level label — a page often spans the tail of one subsection and the start of the next.
3. **Lazy chunking** — chunks are only computed when actually needed (building or rebuilding an index); a cache hit against an existing index skips chunking entirely. `RecursiveCharacterTextSplitter` (`CHUNK_SIZE=900`, `CHUNK_OVERLAP=300`, `add_start_index=True`) splits pages into chunks. `section_for_chunk()` then resolves each chunk's actual section from its own start offset (via the checkpoints from step 2, not the whole page's last heading), and that breadcrumb gets prepended to the chunk's text as `[Section: ...]` before embedding — this is baked into the embedded content, not just display metadata.
4. **Per-PDF vector store caching** — `vector_store_path_for()` derives a unique index directory under `faiss_indexes/<pdf-stem>_<hash>` from `filename|size|mtime`, so different PDFs (or edited versions of the same PDF) never collide or reuse a stale index. If an index directory exists, it's loaded and sanity-checked (`stored_source` metadata must match the current PDF's filename); otherwise a new FAISS index is built and saved there. The legacy top-level `faiss_index/` directory is unused dead weight from an earlier version (see commented-out code) — new indexes always go under `faiss_indexes/`.
5. **Retrieval, diversification, and reranking** — `vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})` (k=40) pulls candidate chunks per question. `diversify_by_section()` caps candidates to at most 3 per section so one dominant section can't crowd out others on broad questions. `rerank_documents()` then asks the LLM to output an ordered list of chunk numbers and actually reorders by that ranking (parsed via regex, deduped, falls back to original order if parsing fails) — keeping the top `TOP_N_CHUNKS` (8).
6. **Answering** — the top reranked docs are passed directly into `qa_chain.combine_documents_chain.invoke()` (bypassing the chain's own retriever, since retrieval/reranking already happened). `qa_chain` is built with a custom `qa_prompt` (via `chain_type_kwargs`) instructing the model to preserve exact conditional language from the source ("the greater of X or Y", not just "X") and to admit when the context doesn't contain an answer rather than guess. Answers are printed along with a deduplicated `Sources: Page X, Page Y` line (1-indexed — `PyPDFLoader`'s raw `page` metadata is 0-indexed) and a per-chunk source preview (source filename + page + snippet).
7. **Question loop with follow-up resolution** — a `while True` REPL reads from `input()` until the user types `exit`. Before each question is used for retrieval, `condense_question()` checks `chat_history` (last `CHAT_HISTORY_TURNS`, default 5, question/answer pairs) and — only if history exists — asks the LLM to rewrite a follow-up like "what about X?" into a standalone question. This condensed question drives retrieval and reranking; the actual answer generation step still only sees retrieved document context, never the raw chat history, so answers stay grounded in the PDF rather than drifting into free-form conversation.

## Key conventions / gotchas

- FAISS indexes are loaded with `allow_dangerous_deserialization=True` — only ever load indexes generated by this script from trusted local PDFs.
- `faiss_indexes/` contains generated artifacts tied to specific PDFs by content hash — don't hand-edit them. Any change to chunking, section-tagging, or the embedding model invalidates existing indexes (they'll error on load with a FAISS dimension mismatch, or silently serve stale content) — delete the relevant `faiss_indexes/<pdf>_<hash>/` directory to force a rebuild.
- The vector store cache key is based on filename + size + mtime, not file content — renaming/touching a PDF without changing its content will cause a rebuild; two different PDFs that happen to share all three will collide.
- Conversation memory only steers retrieval (via `condense_question`); it's intentionally not fed into `qa_prompt` directly, to keep answers grounded in retrieved document text rather than the model's own recollection of the conversation.
- Chat history is per-process/per-session — it isn't persisted across runs.
- Known limitation: the local LLM (llama3, 8B) sometimes fails to reliably extract facts from linearized PDF tables (e.g. connecting a tenure tier like "Year 11 and thereafter" to its own accrual figure a line later) — this is a model reasoning ceiling, not a retrieval/chunking bug, and isn't reliably fixable by tuning chunk parameters further.

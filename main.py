#LangChain literally restricts the LLM to only use retrieved context — not general knowledge.

# STEP 1: Import all necessary libraries
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
import os
from langchain.prompts import PromptTemplate
import hashlib
import pathlib
import re

TOP_N_CHUNKS = 8
CHUNK_SIZE = 900
CHUNK_OVERLAP = 300
RETRIEVAL_K = 40
LLM_MODEL = "llama3"
# llama3 only declares the "completion" capability in Ollama, not "embedding" —
# it can't serve embeddings. Use a dedicated embedding model instead.
EMBEDDING_MODEL = "nomic-embed-text"

#from langchain_core.runnables import RunnableSequence

def load_user_pdf():
    import tkinter as tk
    from tkinter import filedialog
    from langchain_community.document_loaders import PyPDFLoader

    root = tk.Tk()
    root.withdraw()  # hide the empty Tk window

    file_path = filedialog.askopenfilename(
        title="Select a PDF",
        filetypes=[("PDF files", "*.pdf")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        raise SystemExit(0)

    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f" Loaded {len(pages)} pages from: {file_path}")
    return pages, file_path

#pdf_path = "document.pdf"

#FAISS->vector search engine
#stores all chunks as vectors and when we ask ques it finds the most similar chunks
#vector_store=FAISS.from_documents(chunks,embeddings)
#VECTOR_DB_PATH = "faiss_index"
#
#if os.path.exists(VECTOR_DB_PATH):
#    print(" Loading saved vector store...")
#    vector_store = FAISS.load_local(
#        folder_path=VECTOR_DB_PATH,
#        embeddings=embeddings,
#        allow_dangerous_deserialization=True
#    )
#else:
#    print(" Creating new vector store from PDF chunks...")
#    vector_store = FAISS.from_documents(chunks, embeddings)
#    vector_store.save_local(VECTOR_DB_PATH)
#    print(" Saved vector store for future use.")

def vector_store_path_for(pdf_path: str) -> str:
    """
    Unique dir per PDF (filename + size + mtime) so different PDFs never collide.
    """
    p = pathlib.Path(pdf_path)
    key = f"{p.name}|{p.stat().st_size}|{int(p.stat().st_mtime)}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    base = "faiss_indexes"
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{p.stem}_{h}")

def build_faiss(chunks, embeddings, index_dir):
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(index_dir)
    return vs

# Best-effort heading detection for numbered/lettered documents (I., A., 1., ...).
# Tagging pages with their section breadcrumb helps retrieval surface chunks from
# less-obviously-matching sections (e.g. "Vacation Benefits") when a broad question
# ("amount of leave") would otherwise be dominated by whichever section's wording
# happens to overlap the query most.
_HEADING_PATTERNS = [
    re.compile(r"^(?P<num>[IVXLCDM]{1,6})\.\s+(?P<title>[A-Z][A-Z ,&/\-]{3,80})$"),
    re.compile(r"^(?P<num>[A-Z])\.\s+(?P<title>[A-Z][\w ,&/()'\-]{2,80})$"),
    re.compile(r"^(?P<num>\d{1,2})\.\s+(?P<title>[A-Z][\w ,&/()'\-]{2,80})$"),
]

def _detect_heading(line):
    line = line.strip()
    if not line or len(line) > 90:
        return None
    for level, pattern in enumerate(_HEADING_PATTERNS):
        m = pattern.match(line)
        if m:
            return level, f"{m.group('num')}. {m.group('title').strip()}"
    return None

def tag_pages_with_sections(pages):
    """Record, per page, the section active on entry plus an offset-ordered list
    of headings found mid-page. A single page often spans multiple subsections
    (e.g. the tail of one leave policy and the start of the next), so tagging
    the whole page with only its last heading would mislabel earlier chunks."""
    stack = [None, None, None]
    for page in pages:
        page.metadata["_section_entry"] = " > ".join(s for s in stack if s)
        checkpoints = []
        offset = 0
        for line in page.page_content.splitlines(keepends=True):
            hit = _detect_heading(line.rstrip("\n"))
            if hit is not None:
                level, label = hit
                stack[level] = label
                for deeper in range(level + 1, len(stack)):
                    stack[deeper] = None
                checkpoints.append((offset, " > ".join(s for s in stack if s)))
            offset += len(line)
        page.metadata["_section_checkpoints"] = checkpoints

def section_for_chunk(chunk):
    """Resolve the section active at this chunk's actual start offset within
    its source page, using the checkpoints tag_pages_with_sections recorded."""
    section = chunk.metadata.get("_section_entry", "")
    start = chunk.metadata.get("start_index", 0)
    for offset, label in chunk.metadata.get("_section_checkpoints", []):
        if offset <= start:
            section = label
        else:
            break
    return section

def diversify_by_section(docs, max_per_section=3):
    """Cap how many retrieved candidates come from the same section so one
    dominant section doesn't crowd out other equally relevant ones before reranking."""
    counts = {}
    kept = []
    for doc in docs:
        key = doc.metadata.get("section") or doc.metadata.get("source", "")
        if counts.get(key, 0) >= max_per_section:
            continue
        counts[key] = counts.get(key, 0) + 1
        kept.append(doc)
    return kept

rerank_prompt = PromptTemplate.from_template(
    """
    You are ranking text chunks by how useful they are for answering a question.

    Question:
    {question}

    Chunks:
    {context}

    Reply with ONLY a comma-separated list of chunk numbers, most relevant first
    (e.g. "3,1,4,2"). Do not include any other text.
    """
)

def rerank_documents(llm, rerank_prompt, question, docs):
    """Ask the LLM to rank the candidate chunks and reorder them accordingly."""
    context = "\n\n".join([f"[{i}] {d.page_content[:500]}" for i, d in enumerate(docs)])
    rerank_input = rerank_prompt.format(question=question, context=context)
    ranked_text = llm.invoke(rerank_input)

    ordered_indices = []
    seen = set()
    for match in re.findall(r"\d+", ranked_text):
        idx = int(match)
        if 0 <= idx < len(docs) and idx not in seen:
            ordered_indices.append(idx)
            seen.add(idx)

    if not ordered_indices:
        # LLM reply didn't parse into usable indices — fall back to original order.
        return docs[:TOP_N_CHUNKS]

    ranked_docs = [docs[i] for i in ordered_indices]
    return ranked_docs[:TOP_N_CHUNKS]

qa_prompt = PromptTemplate.from_template(
    """
    You are answering questions using ONLY the provided context from a PDF document.

    Rules:
    - Answer strictly using the context below; do not use outside knowledge.
    - Preserve exact numbers, conditions, and qualifiers from the context verbatim
      (e.g. if the context says "the greater of X or Y", say "the greater of X or Y",
      not just "X"). Do not round, simplify, or drop conditional language.
    - If the context does not contain the answer, say so explicitly instead of guessing.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

def format_sources(source_pages):
    if not source_pages:
        return "Sources: None"
    unique_pages = sorted(set(source_pages))
    page_list = ", ".join(f"Page {p}" for p in unique_pages)
    return f"Sources: {page_list}"

def main():
    pages, pdf_path = load_user_pdf()
    tag_pages_with_sections(pages)

    # convert chunks into vector embeddings
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    INDEX_DIR = vector_store_path_for(pdf_path)
    print(f" Index dir for this PDF: {INDEX_DIR}")

    #LLMs(LLaMA3) can't handle 50 pages at once so we break it down — only do this
    #if we actually need to build/rebuild the index; a cached hit skips it entirely.
    _chunks_cache = {}
    def get_chunks():
        if "chunks" not in _chunks_cache:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
            )
            chunks = splitter.split_documents(pages)
            for chunk in chunks:
                section = section_for_chunk(chunk)
                if section:
                    chunk.metadata["section"] = section
                    chunk.page_content = f"[Section: {section}]\n{chunk.page_content}"
                chunk.metadata.pop("_section_checkpoints", None)
                chunk.metadata.pop("_section_entry", None)
            _chunks_cache["chunks"] = chunks
            print(f"split into {len(chunks)} chunks.")
        return _chunks_cache["chunks"]

    if os.path.exists(INDEX_DIR):
        print(" Loading saved vector store...")
        vector_store = FAISS.load_local(
            folder_path=INDEX_DIR,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        # sanity check: verify stored source matches current pdf
        try:
            any_id = next(iter(vector_store.docstore._dict))
            any_doc = vector_store.docstore._dict[any_id]
            stored_source = any_doc.metadata.get("source", "")
            if pathlib.Path(stored_source).name != pathlib.Path(pdf_path).name:
                print(" Index/source mismatch. Rebuilding index for this PDF...")
                vector_store = build_faiss(get_chunks(), embeddings, INDEX_DIR)
                print("  Rebuilt and saved vector store.")
        except Exception as e:
            print(f"  Couldn’t validate index ({e}). Rebuilding...")
            vector_store = build_faiss(get_chunks(), embeddings, INDEX_DIR)
            print("  Rebuilt and saved vector store.")
    else:
        print(" Creating new vector store from PDF chunks...")
        vector_store = build_faiss(get_chunks(), embeddings, INDEX_DIR)
        print(" Saved vector store for future use.")

    #connect to local llm using ollama
    llm = OllamaLLM(model=LLM_MODEL)

    # STEP 9: Build QA Chain
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})  # first-pass retrieval count
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt},
    )
    print(" QA Chain ready!")

    # STEP 10: Ask Questions
    print("\n Ask anything about the PDF (type 'exit' to quit):")
    while True:
        question = input("\n You: ")
        if question.lower() == "exit":
            print(" Bye!")
            break

        # Step A: get retrieved docs, capped per section so one dominant
        # section can't crowd out other equally relevant ones
        docs = retriever.invoke(question)
        docs = diversify_by_section(docs)

        # Step B: rerank them
        top_docs = rerank_documents(llm, rerank_prompt, question, docs)

        # Step C: run QA only on top reranked docs
        response = qa_chain.combine_documents_chain.invoke({
            "input_documents": top_docs,
            "question": question
        })
        answer = response["output_text"]

        # Extract page numbers (PyPDFLoader pages are 0-indexed; +1 for display)
        source_pages = [
            doc.metadata.get("page") + 1
            for doc in top_docs
            if isinstance(doc.metadata.get("page", None), int)
        ]
        formatted_sources = format_sources(source_pages)

        print(f"\n Answer:\n{answer}\n\n {formatted_sources}\n")

        print(" Source Preview:")
        for i, doc in enumerate(top_docs):
            page = doc.metadata.get("page", "?")
            page_display = page + 1 if isinstance(page, int) else page
            source = pathlib.Path(doc.metadata.get("source", "?")).name
            snippet = doc.page_content[:200].replace("\n", " ")
            print(f"  • {source} — Page {page_display}: {snippet}...\n")

if __name__ == "__main__":
    main()

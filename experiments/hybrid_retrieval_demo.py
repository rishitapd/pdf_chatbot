"""
DEMO / PROTOTYPE — not the working chatbot. See ../main.py for that.

This is an experiment in hybrid retrieval: combining the FAISS (semantic)
retriever with a BM25 (keyword) retriever via LangChain's EnsembleRetriever,
plus a keyword-based prefilter for policy-heavy questions.

Known issues, kept as-is for reference rather than fixed:
- Uses OllamaEmbeddings(model="llama3") — llama3 only declares the
  "completion" capability in Ollama, not "embedding", so this will fail with
  a 501 error until swapped for a real embedding model (main.py uses
  nomic-embed-text).
- Requires the `rank_bm25` package (`pip install rank_bm25`) for
  BM25Retriever, which isn't in any requirements file.
- `docs_filtered` (the keyword prefilter output) is computed but never
  passed to rerank_documents — only the unfiltered `docs` is used, so the
  prefilter isn't actually wired into the pipeline yet.
- `rerank_documents` still calls the LLM but discards its ranking, same
  issue main.py's reranker had before it was fixed.
"""

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
#for hybrid retrieval
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever



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

def format_sources(source_pages):
    if not source_pages:
        return "Sources: None"
    unique_pages = sorted(set(source_pages))
    page_list = ", ".join(f"Page {p}" for p in unique_pages)
    return f"Sources: {page_list}"

def keyword_prefilter(docs, terms):
    """
    Keep only docs that contain any of the terms. If none match, return original docs.
    Useful for policy/keyword-heavy queries.
    """
    if not docs:
        return docs
    terms = [t.lower() for t in terms]
    filtered = [d for d in docs if any(t in (d.page_content or "").lower() for t in terms)]
    return filtered if filtered else docs

# ---------------- Pipeline ---------------


pages, pdf_path = load_user_pdf()

#pdf_path = "document.pdf"

#LLMs(LLaMA3) can't handle  50 pages at once so we break it down
splitter=RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap=200)
chunks=splitter.split_documents(pages)
print(f"split into {len(chunks)} chunks.")

# convet chunks into vector embeddings
embeddings=OllamaEmbeddings(model="llama3")

INDEX_DIR = vector_store_path_for(pdf_path)
print(f" Index dir for this PDF: {INDEX_DIR}")

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
            vector_store = build_faiss(chunks, embeddings, INDEX_DIR)
            print("  Rebuilt and saved vector store.")
    except Exception as e:
        print(f"  Couldn’t validate index ({e}). Rebuilding...")
        vector_store = build_faiss(chunks, embeddings, INDEX_DIR)
        print("  Rebuilt and saved vector store.")
else:
    print(" Creating new vector store from PDF chunks...")
    vector_store = build_faiss(chunks, embeddings, INDEX_DIR)
    print(" Saved vector store for future use.")



#connect to local llm using ollama
llm=OllamaLLM(model="llama3")
# Build Hybrid retriever: FAISS (semantic) + BM25 (keyword)
# FAISS retriever
faiss_ret = vector_store.as_retriever(search_kwargs={"k": 30, "fetch_k": 30})
# BM25 retriever (requires rank_bm25)
bm25_ret = BM25Retriever.from_documents(chunks)
bm25_ret.k = 30

# Ensemble/hybrid retriever
retriever = EnsembleRetriever(
    retrievers=[faiss_ret, bm25_ret],
    weights=[0.6, 0.4]
)

rerank_prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant for answering questions based only on the provided documents.

    When answering:
    - Rank chunks of text based on how useful they might be for answering the question.
    - Keep chunks if they contain keywords, related concepts, or explanations that could help.
    - At the end, cite the page numbers (e.g., Sources: Page 2, Page 5).

    Context:
    {context}

    Question:
    {question}

    Rank the chunks (most relevant first). Keep useful ones even if they don’t contain the full answer.
    """
)
# STEP 8: Retriever with Reranking
#def rerank_documents(question, docs):
#    context = "\n\n".join([d.page_content[:500] for d in docs])
#    rerank_input = rerank_prompt.format(question=question, context=context)
#    ranked_text = llm.invoke(rerank_input)
#
#    # Simple trick: keep top N (e.g., 5)
#    top_docs = docs[:5]
#    return top_docs

def rerank_documents(question, docs, keep_top_n=5):
    """
    Placeholder reranker: call the LLM for relevance reasoning, but to avoid
    brittle parsing we currently return the top N docs (preserve safety).
    We keep the LLM call so you can iterate to a true parser later.
    """
    context = "\n\n".join([d.page_content[:500] for d in docs])
    _ = llm.invoke(rerank_prompt.format(question=question, context=context))
    return docs[:keep_top_n]

# STEP 9: Build QA Chain
#retriever = vector_store.as_retriever(search_kwargs={"k": 30})  # get 15 first
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)
print(" QA Chain ready!")


# Ask loop
print("\n Ask anything about the PDF (type 'exit' to quit):")
policy_terms = ["harassment", "complaint", "report", "pto", "paid time off", "vacation",
                "sick leave", "holiday", "accrual", "carryover", "discipline", "grievance"]

while True:
    question = input("\n You: ").strip()
    if question.lower() == "exit":
        print(" Bye!")
        break

# STEP 10: Ask Questions
#def format_sources(source_pages):
#    if not source_pages:
#        return "Sources: None"
#    unique_pages = sorted(set(source_pages))
#    page_list = ", ".join(f"Page {p}" for p in unique_pages)
#    return f"Sources: {page_list}"
#
#print("\n Ask anything about the PDF (type 'exit' to quit):")
#while True:
#    question = input("\n You: ")
#    if question.lower() == "exit":
#        print(" Bye!")
#        break
#
    # Step A: get retrieved docs(hybrid)
    docs = retriever.invoke(question)

    # Optional: keyword prefilter to emphasize policy matches
    docs_filtered = keyword_prefilter(docs, policy_terms)

    # Step B: rerank them
    top_docs = rerank_documents(question, docs)

    # Step C: run QA only on top reranked docs
    response = qa_chain.combine_documents_chain.invoke({
        "input_documents": top_docs,
        "question": question

    })
    answer = response.get("output_text", "").strip()

# Extract page numbers
    source_pages = [doc.metadata.get("page", "?") for doc in top_docs if isinstance(doc.metadata.get("page", None), int)]
    formatted_sources = format_sources(source_pages)

    print(f"\n Answer:\n{answer}\n\n {formatted_sources}\n")

# Preview chunks
  #  print(" Source Preview:")
  #  for i, doc in enumerate(top_docs):
  #      page = doc.metadata.get("page", "?")
  #      snippet = doc.page_content[:200].replace("\n", " ")
  #      print(f"  • Page {page}: {snippet}...\n")


    print(" Source Preview:")
    for i, doc in enumerate(top_docs):
        page = doc.metadata.get("page", "?")
        source = pathlib.Path(doc.metadata.get("source", "?")).name
        snippet = doc.page_content[:200].replace("\n", " ")
        print(f"  • {source} — Page {page}: {snippet}...\n")

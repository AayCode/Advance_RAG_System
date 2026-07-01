# Hybrid RAG System

A Retrieval-Augmented Generation (RAG) system that answers questions from your documents using a hybrid search approach — combining semantic vector search with keyword-based BM25 retrieval for more accurate results.

Built with LangChain, FAISS, Sentence Transformers, and Groq LLM, with a Streamlit chat interface.

---

## Features

- **Hybrid Retrieval** — combines FAISS semantic search and BM25 keyword search, merged via a custom scoring function with priority boosting and length-based penalties
- **Query Rewriting** — LLM-based query rewriting that expands and clarifies user queries before retrieval for better accuracy
- **Confidence-Based Source Attribution** — sources are only shown when retrieval confidence exceeds a threshold, reducing misleading citations
- **General Query Handling** — casual/greeting queries bypass RAG entirely and go directly to the LLM
- **Streamlit Chat UI** — interactive multi-turn chat interface with expandable source references

---

## Tech Stack

| Component | Library |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Store | `FAISS` (IndexFlatL2) |
| Keyword Search | `rank-bm25` (BM25Okapi) |
| LLM | `Groq` (llama-3.1-8b-instant) |
| Document Parsing | `pypdf` |
| Frontend | `Streamlit` |

---

## Project Structure

```
├── ingestion/
│   ├── load_docs.py       # PDF loading and text extraction
│   └── chunking.py        # Text splitting with overlap
├── embeddings/
│   └── embedder.py        # Sentence transformer embedding model
├── retrieval/
│   ├── vector_store.py    # FAISS index build and search
│   ├── bm25.py            # BM25 keyword retriever
│   └── hybrid.py          # Hybrid retrieval with scoring and merging
├── rag/
│   ├── query_rewrite.py   # LLM-based query rewriting
│   └── generator.py       # Answer generation with Groq
├── data/
│   └── pdfs/              # Place your PDF documents here
├── app1.py                # Streamlit app entry point
└── .env                   # API keys (never committed)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Aaycode/hybrid-rag-system.git
cd hybrid-rag-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Add your documents

Place your PDF files inside the `data/pdfs/` folder.

### 5. Run the app

```bash
streamlit run app1.py
```

---

## How It Works

1. **Document Ingestion** — PDFs are loaded page by page using `pypdf`
2. **Chunking** — Text is split into 700-character chunks with 150-character overlap using `RecursiveCharacterTextSplitter`
3. **Embedding** — Each chunk is embedded using `all-MiniLM-L6-v2` (fast, lightweight, CPU-friendly)
4. **Indexing** — Embeddings are stored in a FAISS `IndexFlatL2` for exact nearest-neighbor search; BM25 index is built in parallel
5. **Query Rewriting** — Incoming queries are rewritten by the LLM to be more retrieval-friendly
6. **Hybrid Search** — Both FAISS (semantic) and BM25 (keyword) searches are run; results are merged and ranked using a custom scoring function
7. **Generation** — Top chunks are passed as context to the Groq LLM, which generates a grounded answer
8. **Source Display** — If retrieval confidence score exceeds the threshold, source documents and page numbers are shown

---

## Known Limitations

- FAISS `IndexFlatL2` is exact search — works well for small datasets but does not scale to millions of chunks (HNSW/IVF would be better for large corpora)
- Query rewriting adds one extra LLM call per query, slightly increasing response latency
- BM25 tokenization is whitespace-based (no stemming or lemmatization)

---

## Requirements

```
streamlit
langchain-text-splitters
sentence-transformers
faiss-cpu
rank-bm25
pypdf
groq
python-dotenv
numpy
```

---

## License

MIT

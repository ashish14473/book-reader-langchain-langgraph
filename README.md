# 📚 LangGraph RAG Demo — Teaching App

A clean, well-commented demo showing how to build a **Retrieval-Augmented Generation (RAG)** 
pipeline using **LangGraph**, **ChromaDB**, and **Claude**.

---

## 🗂️ Project Structure

```
langgraph-rag-demo/
│
├── rag_graph.py      ← Core pipeline (LangGraph + ChromaDB + Claude)
├── app.py            ← Streamlit UI
├── cli_demo.py       ← Quick CLI test (no UI needed)
├── requirements.txt  ← Python dependencies
└── README.md
```

---

## 🔁 Pipeline Architecture

```
User Query
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                LangGraph RAG Graph                  │
│                                                     │
│   START ──▶ [retrieve] ──▶ [generate] ──▶ END       │
│                  │               │                  │
│           ChromaDB search   Claude API              │
│           (top-k chunks)   (RAG answer)             │
└─────────────────────────────────────────────────────┘
```

### Node 1 — `retrieve`
- Converts user query to embedding vector
- Searches ChromaDB for top-k similar chunks
- Adds chunks to the shared `RAGState`

### Node 2 — `generate`
- Joins chunks into a context string
- Sends `CONTEXT + QUESTION` → Claude
- Claude generates a grounded answer

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3a. Run the Streamlit app (recommended)
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

### 3b. Or use the CLI
```bash
# First time — ingest a PDF
python cli_demo.py --pdf path/to/yourbook.pdf

# Next time — load existing ChromaDB
python cli_demo.py --load
```

---

## 🧠 Key Concepts Taught

| Concept | Where |
|---|---|
| LangGraph `StateGraph` | `rag_graph.py` → `build_rag_graph()` |
| Graph nodes & edges | `rag_graph.py` → `make_retrieve_node()`, `generate()` |
| PDF loading & chunking | `rag_graph.py` → `ingest_pdf()` |
| Vector embeddings | `rag_graph.py` → `HuggingFaceEmbeddings` |
| ChromaDB storage | `rag_graph.py` → `Chroma.from_documents()` |
| Similarity search | `rag_graph.py` → `vectorstore.similarity_search()` |
| Prompt construction | `rag_graph.py` → `generate()` |
| Claude API call | `rag_graph.py` → `anthropic.Anthropic().messages.create()` |

---

## ⚙️ Customization

| Parameter | Default | Where to change |
|---|---|---|
| Chunk size | 1000 chars | `ingest_pdf()` in `rag_graph.py` |
| Chunk overlap | 200 chars | `ingest_pdf()` in `rag_graph.py` |
| Top-k retrieval | 4 chunks | `build_rag_graph(top_k=4)` |
| Embedding model | `all-MiniLM-L6-v2` | `HuggingFaceEmbeddings(model_name=...)` |
| LLM model | `claude-sonnet-4-...` | `generate()` node |

---

## 📦 Technologies Used

- **LangGraph** — Graph-based LLM workflow orchestration
- **LangChain** — PDF loading, text splitting utilities
- **ChromaDB** — Local vector database for storing embeddings
- **HuggingFace Sentence Transformers** — Free local embeddings
- **Anthropic Claude** — LLM for answer generation
- **Streamlit** — Simple Python web UI

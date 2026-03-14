

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
try:
    from rag_graph import ingest_pdf, load_vectorstore, build_rag_graph, ask
    IMPORT_OK = True
    IMPORT_ERROR = None
except Exception as _e:
    IMPORT_OK = False
    IMPORT_ERROR = _e

# ----------------------------------------------------------
#  PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="📚 LangGraph RAG Demo",
    page_icon="📚",
    layout="wide",
)

# Show import errors clearly instead of a blank page
if not IMPORT_OK:
    st.error("❌ **Failed to import dependencies.** See details below.")
    st.code(str(IMPORT_ERROR), language="text")
    st.markdown("### 🔧 Fix: run this in your terminal")
    st.code("pip install -r requirements.txt", language="bash")
    st.stop()   # stop rendering the rest of the app

st.title("📚 LangGraph RAG — Book Q&A Demo")
st.markdown(
    "A teaching demo that shows how **Retrieval-Augmented Generation (RAG)** "
    "works using **LangGraph**, **ChromaDB**, and **Claude**."
)

# ----------------------------------------------------------
#  SIDEBAR — Configuration & PDF Upload
# ----------------------------------------------------------
with st.sidebar:
    
    # PDF Upload
    st.subheader("📄 Upload your PDF Book")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload the book you want to ask questions about",
    )

    # Retrieval settings
    st.divider()
    st.subheader("🔧 Retrieval Settings")
    top_k = st.slider(
        "Chunks to retrieve (top_k)",
        min_value=1,
        max_value=10,
        value=4,
        help="How many text chunks to fetch from the vector store per query",
    )
    chunk_size = st.slider("Chunk size (characters)", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 500, 200, step=50)

    st.divider()
    st.markdown("**Pipeline:**")
    st.markdown("```\nSTART → retrieve → generate → END\n```")


# ----------------------------------------------------------
#  SESSION STATE  –  persists across reruns
# ----------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "graph" not in st.session_state:
    st.session_state.graph = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingested" not in st.session_state:
    st.session_state.ingested = False


# ----------------------------------------------------------
#  STEP 1 — Ingest the PDF
# ----------------------------------------------------------
st.header("Step 1 — Ingest PDF into Vector Store")

col1, col2 = st.columns([3, 1])

with col1:
    if not uploaded_file:
        st.warning("⚠️ Still needed: **PDF file** — upload it in the sidebar.")
    else:
        st.success(f"✅ Ready! **{uploaded_file.name}** ({uploaded_file.size // 1024} KB) — click Ingest PDF →")

with col2:
    ingest_btn = st.button(
        "🚀 Ingest PDF",
        disabled=(not uploaded_file),
        use_container_width=True,
    )

if ingest_btn and uploaded_file:
    # Save uploaded file to a temp path (works on Windows, Mac, Linux)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    with st.spinner("Reading, splitting, embedding, and storing chunks…"):
        try:
            vs = ingest_pdf(temp_path)
            st.session_state.vectorstore = vs
            st.session_state.graph = build_rag_graph(vs, top_k=top_k)
            st.session_state.ingested = True
            st.session_state.chat_history = []  # reset chat on new PDF
            st.success("✅ PDF ingested successfully! You can now ask questions.")
        except Exception as e:
            st.error(f"❌ Ingestion failed: {e}")


# ----------------------------------------------------------
#  STEP 2 — Visual pipeline explainer
# ----------------------------------------------------------
st.header("Step 2 — How the Pipeline Works")

with st.expander("📖 Click to see the LangGraph pipeline diagram", expanded=False):
    st.markdown("""
    ```
    User Query
         │
         ▼
    ┌─────────────────────────────────────────────────────┐
    │                   LangGraph RAG Graph               │
    │                                                     │
    │   ┌──────────┐      ┌──────────┐      ┌─────────┐  │
    │   │  START   │─────▶│ retrieve │─────▶│generate │  │
    │   └──────────┘      └──────────┘      └─────────┘  │
    │                          │                 │        │
    │                    ChromaDB Search    Claude API    │
    │                    (top-k chunks)    (RAG answer)   │
    └─────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                                           Final Answer
    ```

    **Node 1 — `retrieve`**
    - Takes the user query
    - Converts it to an embedding vector
    - Finds the most similar chunks in ChromaDB
    - Returns top-k text chunks

    **Node 2 — `generate`**
    - Combines the retrieved chunks into a context
    - Sends: `CONTEXT + QUESTION` → Claude
    - Claude generates a grounded answer from the book

    **Why LangGraph?**
    - Each node is a pure Python function
    - State flows between nodes automatically
    - Easy to add more nodes (e.g. query rewriting, re-ranking)
    """)


# ----------------------------------------------------------
#  STEP 3 — Q&A Chat Interface
# ----------------------------------------------------------
st.header("Step 3 — Ask Questions About Your Book")

if not st.session_state.ingested:
    st.warning("⚠️ Please ingest a PDF first (Step 1).")
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    query = st.chat_input("Ask a question about the book…")

    if query:
        # Show user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Run the graph
        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving and generating…"):
                try:
                    result = ask(st.session_state.graph, query)
                    answer = result["answer"]

                    st.markdown(answer)

                    # Show retrieved context in an expander (for teaching)
                    with st.expander("🔎 View retrieved chunks from ChromaDB"):
                        for i, chunk in enumerate(result["retrieved_docs"], 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(chunk[:500] + ("…" if len(chunk) > 500 else ""))
                            st.divider()

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    err = f"❌ Error: {e}"
                    st.error(err)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": err}
                    )

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

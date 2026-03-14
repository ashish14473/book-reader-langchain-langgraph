import os
from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

class RAGState(TypedDict):
    """
    The state dictionary flows through every node in the graph.
    Each node can read from it and write new keys into it.
    """
    query: str               # user's question
    retrieved_docs: List[str]  # raw text chunks from the vector store
    context: str             # joined context string sent to GPT
    answer: str              # final answer from GPT


# ----------------------------------------------------------
# 2.  INGESTION  –  PDF → chunks → ChromaDB
#     (Run once to populate the vector store)
# ----------------------------------------------------------
def ingest_pdf(pdf_path: str, collection_name: str = "book_rag") -> Chroma:
    print(f"📄 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()   # returns list of Document objects, one per page
    print(f"   → {len(pages)} pages loaded")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],  # tries these in order
    )
    chunks = splitter.split_documents(pages)
    print(f"   → {len(chunks)} chunks created")

    # ── Embed chunks using OpenAI text-embedding-3-small ─────
    # Fast, cheap, and high quality — great for RAG applications
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    # ── Store in ChromaDB (persisted locally) ────────────────
    print("   → Embedding & storing in ChromaDB …")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_db",   # saved on disk
    )
    print("✅ Ingestion complete!\n")
    return vectorstore


def load_vectorstore(collection_name: str = "book_rag") -> Chroma:
    """
    Loads an already-populated ChromaDB collection from disk.
    Call this instead of ingest_pdf() if you've already ingested the PDF.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )


# ----------------------------------------------------------
# 3.  GRAPH NODES
# ----------------------------------------------------------

def make_retrieve_node(vectorstore: Chroma, top_k: int = 4):
    """
    Factory that creates a 'retrieve' node bound to a vectorstore.

    The node:
      - Takes the user query from state
      - Runs a similarity search in ChromaDB
      - Returns the top-k most relevant text chunks
    """
    def retrieve(state: RAGState) -> dict:
        print(f"\n🔍 [retrieve] Searching for: '{state['query']}'")

        # similarity_search returns Document objects ranked by relevance
        docs = vectorstore.similarity_search(state["query"], k=top_k)

        # Extract plain text from each Document
        texts = [doc.page_content for doc in docs]

        # Show a preview for teaching/debugging
        for i, t in enumerate(texts):
            print(f"   Chunk {i+1}: {t[:120].strip()} …")

        return {"retrieved_docs": texts}

    return retrieve


def generate(state: RAGState) -> dict:
    """
    The 'generate' node:
      - Joins retrieved chunks into a single context string
      - Sends (context + question) to GPT-4o
      - Returns GPT's answer
    """
    print("\n🤖 [generate] Building prompt and calling GPT-4o …")

    # Join all retrieved chunks with a separator
    context = "\n\n---\n\n".join(state["retrieved_docs"])

    # ── Build the prompt ──────────────────────────────────────
    system_prompt = (
        "You are a helpful study assistant. "
        "Answer the user's question using ONLY the context provided below. "
        "If the answer is not in the context, say 'I could not find the answer in the book.' "
        "Be concise and clear."
    )

    user_prompt = (
        f"CONTEXT FROM BOOK:\n{context}\n\n"
        f"QUESTION: {state['query']}\n\n"
        "ANSWER:"
    )

    # ── Call GPT-4o via OpenAI SDK ────────────────────────────
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content
    print(f"   Answer: {answer[:200]} …")

    return {"context": context, "answer": answer}


# ----------------------------------------------------------
# 4.  BUILD THE GRAPH
# ----------------------------------------------------------

def build_rag_graph(vectorstore: Chroma, top_k: int = 4) -> object:
    """
    Assembles the LangGraph RAG pipeline.

    Graph structure:
        START ──▶ retrieve ──▶ generate ──▶ END

    Parameters
    ----------
    vectorstore : ChromaDB vectorstore loaded with book chunks
    top_k       : number of chunks to retrieve per query

    Returns
    -------
    A compiled LangGraph runnable
    """
    # Create a graph that uses RAGState as its shared state
    graph = StateGraph(RAGState)

    # Add nodes (name → function)
    graph.add_node("retrieve", make_retrieve_node(vectorstore, top_k))
    graph.add_node("generate", generate)

    # Add edges (define the flow)
    graph.add_edge(START, "retrieve")   # entry point → retrieve
    graph.add_edge("retrieve", "generate")  # retrieve → generate
    graph.add_edge("generate", END)         # generate → done

    # Compile turns the graph definition into a runnable object
    return graph.compile()


# ----------------------------------------------------------
# 5.  CONVENIENCE FUNCTION  –  one-shot query
# ----------------------------------------------------------

def ask(graph, question: str) -> dict:
    """
    Run the RAG graph for a single question.

    Returns the full final state dict with keys:
        query, retrieved_docs, context, answer
    """
    initial_state: RAGState = {
        "query": question,
        "retrieved_docs": [],
        "context": "",
        "answer": "",
    }
    return graph.invoke(initial_state)

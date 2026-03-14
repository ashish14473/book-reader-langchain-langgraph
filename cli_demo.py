# ============================================================
#  cli_demo.py  –  Quick CLI test (no Streamlit needed)
# ============================================================
#  Usage:
#      python cli_demo.py --pdf path/to/book.pdf
#      python cli_demo.py --load          (if already ingested)
# ============================================================

import argparse
import os
from rag_graph import ingest_pdf, load_vectorstore, build_rag_graph, ask

load_dotenv()
def main():
    parser = argparse.ArgumentParser(description="LangGraph RAG CLI Demo")
    parser.add_argument("--pdf", help="Path to PDF file to ingest")
    parser.add_argument("--load", action="store_true", help="Load existing ChromaDB")
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ Set your ANTHROPIC_API_KEY environment variable first.")
        print("   export ANTHROPIC_API_KEY=sk-ant-...")
        return

    # Load or ingest
    if args.pdf:
        vs = ingest_pdf(args.pdf)
    elif args.load:
        print("📦 Loading existing ChromaDB …")
        vs = load_vectorstore()
        print("✅ Loaded!\n")
    else:
        print("❌ Provide --pdf <path> or --load")
        return

    # Build the LangGraph RAG graph
    graph = build_rag_graph(vs, top_k=4)

    print("=" * 55)
    print("  📚 LangGraph RAG — CLI Demo")
    print("  Type your question. Press Ctrl+C to quit.")
    print("=" * 55)

    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            if not question:
                continue

            result = ask(graph, question)

            print("\n" + "─" * 50)
            print("💬 Answer:")
            print(result["answer"])
            print("─" * 50)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()

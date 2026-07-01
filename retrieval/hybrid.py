from retrieval.vector_store import VectorStore
from retrieval.bm25 import BM25Retriever
from ingestion.load_docs import load_all_pdfs
from ingestion.chunking import chunk_documents
from rag.query_rewrite import rewrite_query
from rag.generator import generate_answer, generate_general_answer


# ✅ Detect general / casual queries
def is_general_query(query):
    casual_queries = [
        "hi", "hello", "hey", "thanks", "thank you",
        "how are you", "good morning", "good evening"
    ]

    query_lower = query.lower().strip()

    # Exact match OR very short query
    return query_lower in casual_queries or len(query_lower.split()) <= 2


class HybridRetriever:
    def __init__(self):
        print("🔄 Loading and processing documents...")

        docs = load_all_pdfs("data/pdfs")
        self.chunks = chunk_documents(docs)

        print(f"📄 Total chunks: {len(self.chunks)}")

        # ✅ Vector store
        self.vector_store = VectorStore()
        self.vector_store.build_index(self.chunks)

        # ✅ BM25
        self.bm25 = BM25Retriever(self.chunks)

        print("✅ Retriever ready!\n")

    def search(self, query, k=5):
        # ✅ Safer query rewrite
        improved_query = rewrite_query(query)
        if not improved_query or len(improved_query.strip()) < 3:
            improved_query = query

        vector_results = self.vector_store.search(improved_query, k=10)
        bm25_results = self.bm25.search(improved_query, k=10)

        scores = {}

        # 🔥 Priority boost
        def get_priority_boost(text):
            text_lower = text.lower()

            if "introduction" in text_lower or "objective" in text_lower:
                return 8
            if "scheme" in text_lower or "overview" in text_lower:
                return 5
            if "fellowship" in text_lower:
                return 2

            return 0

        # 🔥 Length penalty
        def length_penalty(text):
            return min(len(text) / 1000, 3)

        # ✅ Score vector results
        for rank, chunk in enumerate(vector_results):
            key = (chunk["source"], chunk["page"], chunk["chunk_id"])
            base_score = (10 - rank)
            boost = get_priority_boost(chunk["text"])
            penalty = length_penalty(chunk["text"])

            scores[key] = scores.get(key, 0) + base_score + boost - penalty

        # ✅ Score BM25 results
        for rank, chunk in enumerate(bm25_results):
            key = (chunk["source"], chunk["page"], chunk["chunk_id"])
            base_score = (10 - rank)
            boost = get_priority_boost(chunk["text"])
            penalty = length_penalty(chunk["text"])

            scores[key] = scores.get(key, 0) + base_score + boost - penalty

        # ✅ Combine & deduplicate
        all_chunks = vector_results + bm25_results
        unique_chunks = {}

        for chunk in all_chunks:
            key = (chunk["source"], chunk["page"], chunk["chunk_id"])
            unique_chunks[key] = chunk

        # ✅ Sort safely
        sorted_chunks = sorted(
            unique_chunks.values(),
            key=lambda x: scores.get(
                (x["source"], x["page"], x["chunk_id"]), 0
            ),
            reverse=True
        )

        return sorted_chunks[:k], scores


# 🚀 MAIN EXECUTION
if __name__ == "__main__":
    retriever = HybridRetriever()

    while True:
        query = input("Ask a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            print("👋 Exiting...")
            break

        # 🚫 General queries → no RAG
        if is_general_query(query):
            answer = generate_general_answer(query)

            print("\n=== FINAL ANSWER ===\n")
            print(answer)

        # 📄 Document-based queries → RAG
        else:
            results, scores = retriever.search(query)

            answer = generate_answer(query, results)

            print("\n=== FINAL ANSWER ===\n")
            print(answer)

            # 🔥 Confidence-based source display
            if results:
                top = results[0]
                top_score = scores.get(
                    (top["source"], top["page"], top["chunk_id"]), 0
                )

                if top_score > 8:
                    print("\n=== SOURCES ===\n")
                    for r in results:
                        print(f"{r['source']} (Page {r['page']})")
                else:
                    print("\n(No strong document match found)")
import faiss
import numpy as np
from embeddings.embedder import Embedder
from ingestion.chunking import chunk_documents
from ingestion.load_docs import load_all_pdfs


class VectorStore:
    def __init__(self):
        self.embedder = Embedder()
        self.index = None
        self.chunks = []

    def build_index(self, chunks):
        texts = [chunk["text"] for chunk in chunks]

        print("Generating embeddings...")
        embeddings = self.embedder.embed_texts(texts)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))

        self.chunks = chunks
        print("FAISS index built successfully!")

    def search(self, query, k=5):
        query_vector = self.embedder.embed_query(query)

        distances, indices = self.index.search(
            np.array([query_vector]), k
        )

        results = []
        for idx in indices[0]:
            results.append(self.chunks[idx])

        return results


if __name__ == "__main__":
    docs = load_all_pdfs("data/pdfs")
    chunks = chunk_documents(docs)

    vs = VectorStore()
    vs.build_index(chunks)

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        results = vs.search(query)

        print("\nTop Results:\n")
        for i, res in enumerate(results):
            print(f"Result {i+1}:")
            print("Source:", res["source"])
            print("Page:", res["page"])
            print(res["text"][:200])
            print("-" * 50)
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self):
        # Lightweight + good quality model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_texts(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, query):
        return self.model.encode([query])[0]
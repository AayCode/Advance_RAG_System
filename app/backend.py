from fastapi import FastAPI
from pydantic import BaseModel

from retrieval.hybrid import HybridRetriever, is_general_query
from rag.generator import generate_answer, generate_general_answer

app = FastAPI()

retriever = HybridRetriever()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
def chat(req: QueryRequest):
    query = req.query

    if is_general_query(query):
        answer = generate_general_answer(query)
        return {"answer": answer, "sources": []}

    results = retriever.search(query)
    answer = generate_answer(query, results)

    sources = [
        f"{r['source']} (Page {r['page']})"
        for r in results
    ]

    return {
        "answer": answer,
        "sources": sources
    }
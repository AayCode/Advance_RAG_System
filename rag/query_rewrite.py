from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def rewrite_query(query):
    """
    Rewrite the user's query to improve document retrieval.
    Keep the meaning unchanged.
    Expand abbreviations and clarify vague questions.
    """

    try:
        prompt = f"""
You are a query rewriting assistant for a Retrieval-Augmented Generation (RAG) system.

Your task is ONLY to rewrite the user's query so that it is easier to retrieve relevant documents.

Rules:
- Keep the original meaning.
- Do NOT answer the question.
- Expand abbreviations if possible.
- Clarify vague wording.
- Make the query self-contained.
- Return ONLY the rewritten query.

User Query:
{query}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=100,
        )

        rewritten = response.choices[0].message.content.strip()

        print(f"Original Query : {query}")
        print(f"Rewritten Query: {rewritten}")

        return rewritten

    except Exception as e:
        print(f"Query rewrite failed: {e}")
        return query  # Fallback to original query
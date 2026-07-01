import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def classify_query(query):
        query = query.lower()

        if any(word in query for word in ["hi", "hello", "hey"]):
            return "greeting"

        elif any(word in query for word in ["what", "define", "meaning"]):
            return "definition"

        elif any(word in query for word in ["list", "fields", "types"]):
            return "list"

        elif any(word in query for word in ["why", "how"]):
            return "explanation"

        return "general"

def generate_general_answer(query):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # or your working model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    )

    return response.choices[0].message.content

def generate_answer(query, chunks):
    # Combine retrieved context
    query_type = classify_query(query)

    top_chunks = chunks[:3]

    context = ""
    for i, c in enumerate(top_chunks):
        context += f"[{i+1}] Source: {c['source']} (Page {c['page']})\n{c['text']}\n\n"

    prompt = f"""
You are an AI assistant.

Use the provided context to answer the question.

If the answer is clearly found in the context:
→ Answer using ONLY the context.

If the answer is NOT found in the context:
→ Answer using your general knowledge.

Context:
{context}

Question:
{query}

Answer clearly and concisely:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
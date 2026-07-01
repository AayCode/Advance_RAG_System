from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.load_docs import load_all_pdfs


def clean_text(text):
    # Remove page artifacts
    text = text.replace("\n", " ")
    text = text.replace("|", " ")
    
    # Remove extra spaces
    text = " ".join(text.split())

    return text


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )

    all_chunks = []

    for doc in documents:
        chunks = splitter.split_text(doc["text"])

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": clean_text(chunk),
                "source": doc["source"],
                "page": doc["page"],
                "chunk_id": i
            })

    return all_chunks


if __name__ == "__main__":
    docs = load_all_pdfs("data/pdfs")
    chunks = chunk_documents(docs)

    print(f"\nTotal chunks created: {len(chunks)}\n")

    # Show sample chunk
    sample = chunks[0]
    print("Sample Chunk:\n")
    print("Source:", sample["source"])
    print("Page:", sample["page"])
    print("Chunk ID:", sample["chunk_id"])
    print(sample["text"][:500])
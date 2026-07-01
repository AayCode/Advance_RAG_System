import os
from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    pages = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({
                "text": text,
                "page": page_num + 1
            })

    return pages


def load_all_pdfs(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(folder_path, file)
            pages = load_pdf(full_path)

            for page in pages:
                documents.append({
                    "text": page["text"],
                    "source": file,
                    "page": page["page"]
                })

    return documents


if __name__ == "__main__":
    docs = load_all_pdfs("data/pdfs")

    print(f"\nTotal pages loaded: {len(docs)}\n")

    # Show sample
    print("Sample document:\n")
    print("Source:", docs[0]["source"])
    print("Page:", docs[0]["page"])
    print(docs[0]["text"][:500])
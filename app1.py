import streamlit as st
from retrieval.hybrid import HybridRetriever, is_general_query
from rag.generator import generate_answer, generate_general_answer

# ---------------- FILE UPLOAD ----------------

uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        st.write(f"Uploaded: {file.name}")

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Document Assistant",
    layout="wide"
)

st.title("📄 AI Document Assistant")

# ---------------- LOAD RETRIEVER ----------------

@st.cache_resource
def load_retriever():
    return HybridRetriever()

retriever = load_retriever()

# ---------------- MEMORY ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SHOW OLD CHAT ----------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])

        # Show old sources if available
        if msg.get("sources"):

            with st.expander("📚 Sources"):

                for i, src in enumerate(msg["sources"]):

                    st.write(
                        f"[{i+1}] {src['source']} "
                        f"(Page {src['page']})"
                    )

# ---------------- CHAT INPUT ----------------

query = st.chat_input("Ask your question...")

if query:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Show user message
    with st.chat_message("user"):
        st.write(query)

    # ---------------- ASSISTANT RESPONSE ----------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            # General chat
            if is_general_query(query):

                answer = generate_general_answer(query)

                results = []

            # RAG search
            else:

                results, scores = retriever.search(query)

                answer = generate_answer(query, results)

            # Show answer
            st.write(answer)

            # Show sources only if available
            if results:

                st.markdown("---")

                with st.expander("📚 Sources"):

                    for i, res in enumerate(results[:3]):

                        st.write(
                            f"[{i+1}] {res['source']} "
                            f"(Page {res['page']})"
                        )

    # ---------------- SAVE ASSISTANT MESSAGE ----------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": results[:3] if results else []
    })
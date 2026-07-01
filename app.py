import streamlit as st
from retrieval.hybrid import HybridRetriever, is_general_query
from rag.generator import generate_answer, generate_general_answer

uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        st.write(f"Uploaded: {file.name}")

st.set_page_config(page_title="AI Document Assistant", layout="wide")

st.title("📄 AI Document Assistant")

# Load retriever once
@st.cache_resource
def load_retriever():
    return HybridRetriever()

retriever = load_retriever()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
query = st.chat_input("Ask your question...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if is_general_query(query):
                answer = generate_general_answer(query)
                results = None
            else:
                results, scores = retriever.search(query)
                answer = generate_answer(query, results)

            st.write(answer)

            # Show sources
            st.markdown("---")
            with st.expander("📚 Sources"):
                if results:
                    st.markdown("---")
                    st.subheader("📚 Sources")

                    for i, res in enumerate(results[:3]):
                        st.write(f"[{i+1}] {res['source']} (Page {res['page']})")
            

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": results})
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.title("📄 AI Document Assistant")

query = st.text_input("Ask something")

if query:
    response = requests.post(API_URL, json={"query": query})
    data = response.json()

    st.write(data["answer"])

    if data["sources"]:
        st.subheader("📚 Sources")
        for s in data["sources"]:
            st.write(f"{s['source']} (Page {s['page']})")
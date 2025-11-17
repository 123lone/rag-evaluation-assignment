import streamlit as st
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from assignment_1.rag_pipeline import generate_rag_answer

st.set_page_config(page_title="Groq RAG System", layout="wide")
st.title("⚡ RAG System using Groq Llama-3.1-70B")

query = st.text_input("Enter your question:")

if st.button("Search") and query:
    answer, context = generate_rag_answer(query)

    st.subheader("🔍 Retrieved Context")
    st.write(context)

    st.subheader("💡 RAG Answer")
    st.write(answer)

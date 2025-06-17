import streamlit as st
from parser import extract_text_from_pdf
from rag_engine import build_vector_store
from chatbot import build_qa_chain

st.set_page_config(page_title="Restaurant Chatbot Demo", layout="centered")

st.title("🍕 Restaurant Chatbot Demo")

uploaded_file = st.file_uploader("Upload a menu (PDF only)", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from menu..."):
        raw_text = extract_text_from_pdf(uploaded_file)

    with st.spinner("Building search index..."):
        vector_store = build_vector_store(raw_text)
        qa_chain = build_qa_chain(vector_store)

    st.success("Chatbot is ready! Ask about the menu below.")

    query = st.text_input("Ask a question (e.g., 'What gluten-free options do you have?')")

    if query:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(query)
            st.markdown(f"**Answer:** {answer}")

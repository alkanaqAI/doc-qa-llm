import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import tempfile
import os
import subprocess
import requests
import time

def ensure_ollama_running():
    try:
        requests.get("http://localhost:11434")
    except requests.exceptions.ConnectionError:
        subprocess.Popen(["ollama", "serve"])
        for _ in range(10):
            try:
                requests.get("http://localhost:11434")
                break
            except requests.exceptions.ConnectionError:
                time.sleep(1)

st.title("🧠 PDF Question Answering (Local LLM)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Enter your question:")

if uploaded_file and question:
    with st.spinner("Processing document..."):

        # Save uploaded PDF to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        ensure_ollama_running()

        # Load and split
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        # Embed & retrieve
        embeddings = OllamaEmbeddings(model="mistral")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()
        llm = ChatOllama(model="mistral")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Run QA
        response = qa_chain.invoke({"query": question})

        st.markdown("### 💬 Answer:")
        st.write(response["result"])

        os.remove(pdf_path)

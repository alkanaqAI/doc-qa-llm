import os
import subprocess
import requests
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS  # This was missing

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

ensure_ollama_running()  # 🔑 Must come early, before model/embedding usage

# Load env vars
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
assert openai_key, "❌ OPENAI_API_KEY not found in .env"

# Load & split PDF
pdf_path = "data/nature_paper.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

print(f"✅ Loaded {len(docs)} chunks from {pdf_path}")
print("--- Sample chunk ---")
print(docs[0].page_content)

# Use local Mistral model via Ollama
embedding_model = OllamaEmbeddings(model="mistral")
vectorstore = FAISS.from_documents(docs, embedding_model)

retriever = vectorstore.as_retriever()
llm = ChatOllama(model="mistral")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Test query
response = qa_chain.invoke({"query":"What is the main hypothesis of the paper?"})
print(f"\n🧠 Response:\n{response}")

# Save vector index
faiss_path = "data/faiss_index"
vectorstore.save_local(faiss_path)
print(f"✅ FAISS index saved at: {faiss_path}")

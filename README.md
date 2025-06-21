# Local Document QA with LLM (Mistral via Ollama)

This project demonstrates a simple yet powerful pipeline to perform question-answering over a local PDF document using a local LLM (Mistral) via [Ollama](https://ollama.com). It also supports switching to OpenAI's API if desired.

---

## 🔧 Features

- Load and chunk PDF documents
- Generate embeddings using `OllamaEmbeddings` (Mistral)
- Store/retrieve chunks with FAISS vectorstore
- Run local LLM (`mistral`) for question-answering using LangChain
- Automatically starts Ollama server if not already running
- Supports `ChatOllama` and `OpenAIEmbeddings` (if key provided)

---

## 🧠 Requirements

- macOS with Apple Silicon (tested on M1 Max)
- Python 3.11+
- [Ollama](https://ollama.com) installed (`brew install ollama`)
- Mistral model pulled: `ollama pull mistral`

---

## 📁 Project Structure
```
doc-qa-llm/
├── data/
│   └── nature_paper.pdf       # Your input PDF
│   └── faiss_index/           # Saved vectorstore
├── scripts/
│   └── process_document.py    # Main logic
├── .env                       # Contains your OPENAI_API_KEY (optional)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
```
---

## ▶️ Running the Pipeline

1. **Clone repo** and install deps:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. **Pull the mistral model:**
   ollama pull mistral

3. **Run the script:**
   python scripts/process_document.py

--- 

✅ Output

The script will:
	•	Start Ollama if not running
	•	Load and chunk your PDF
	•	Embed chunks with Mistral
	•	Store vectors in FAISS
	•	Run a sample QA:
“What is the main hypothesis of the paper?”

⸻

🌐 Optional: Use OpenAI API Instead

To use OpenAI instead of local LLM:
	•	Install openai and langchain-openai
	•	Replace OllamaEmbeddings/ChatOllama with OpenAIEmbeddings and ChatOpenAI
	•	Make sure .env has your OPENAI_API_KEY

⸻

🖥️ Next Steps
	•	Add a Streamlit or FastAPI frontend
	•	Allow uploading different PDFs dynamically
	•	Build a UI for interactive chat with context

⸻

📌 Notes
	•	You do not need an internet connection to run the Mistral model after pulling it once
	•	This is a great example of using agentic AI tools locally

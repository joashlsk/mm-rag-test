# mm-rag-test
streamlit (Front end UI)
FAISS (Facebook AI Similarity Search)


rag-system-root/
├── frontend/               # Streamlit App (User Interface)
│   ├── app.py              # Main UI Entry Point (Chat & Sidebar)
│   └── components.py       # UI helpers (e.g., rendering chat messages)
│
├── backend/                # Logic & API (The Brain)
│   ├── main.py             # API Entry Point (FastAPI app setup)
│   ├── api/                # Endpoints / Routes
│   │   ├── chat.py         # The /chat endpoint (Handles querying Groq)
│   │   └── document.py     # The /upload endpoint (Handles file processing)
│   │
│   ├── core/               # Core configuration and clients
│   │   ├── config.py       # Environment variables (GROQ_API_KEY loading)
│   │   └── llm_client.py   # Groq API initialization
│   │
│   └── services/           # RAG Business Logic
│       ├── ingestion.py    # PyPDF reading & LangChain text chunking
│       ├── embeddings.py   # FastEmbed initialization 
│       └── retrieval.py    # FAISS indexing and similarity search logic
│
├── data/                   # Data Storage Layer
│   ├── uploads/            # Temporary folder for uploaded PDFs and TXTs
│   └── faiss_index/        # Directory to save/load persistent FAISS vectors
│
├── .env                    # Your secrets file (e.g., GROQ_API_KEY=...)
├── .gitignore              # Ignores .env and data/ folders from Git
└── requirements.txt        # Complete list of Python dependencies

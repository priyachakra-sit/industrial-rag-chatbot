# ⚡ InsightForge AI

An Industrial AI Analytics Workspace powered by RAG, FAISS, Groq LLM, Tavily Search, and Streamlit.

InsightForge AI enables users to chat with AI, analyze reports, upload documents, generate insights, detect anomalies, and perform intelligent document-based question answering.

---

## 🚀 Features

### 🧠 General AI Chat
- Conversational AI Assistant
- Context-aware memory
- Groq LLM powered responses
- Human-like interaction

### 🌐 Web Search Integration
- Tavily Search API
- Latest information retrieval
- Current trends and news support

### 📂 Intelligent Document Analysis
Upload:

- PDF Reports
- Excel Files (.xlsx)
- CSV Files

Then:

- Ask questions about documents
- Generate summaries
- Extract insights
- Detect anomalies
- Analyze datasets

### 🔍 RAG Pipeline
- Document Chunking
- Embedding Generation
- FAISS Vector Database
- Semantic Search
- Context Retrieval

### 📊 Analytics Engine
Automatically:

- Counts rows & columns
- Detects missing values
- Generates descriptive statistics
- Produces actionable insights

### 🎨 Premium UI
- Glassmorphism design
- Responsive layout
- Modern AI workspace experience

---

# 🏗 Architecture

User
│
▼
Streamlit UI
│
├── General Chat
│ └── Groq LLM
│
├── Web Search
│ └── Tavily API
│
└── Document Upload
│
├── PDF Loader
├── Excel Reader
├── CSV Reader
│
▼
Text Splitter
│
▼
HuggingFace Embeddings
│
▼
FAISS Vector Store
│
▼
Semantic Retrieval
│
▼
Groq LLM
│
▼
AI Response

---

# 🛠 Tech Stack

| Technology | Purpose |
|------------|----------|
| Streamlit | Frontend UI |
| Groq | LLM Inference |
| Tavily | Web Search |
| LangChain | AI Orchestration |
| HuggingFace Embeddings | Vector Embeddings |
| FAISS | Vector Database |
| PyPDF | PDF Processing |
| Pandas | Data Analysis |
| Plotly | Data Visualization |
| OpenPyXL | Excel Processing |

---


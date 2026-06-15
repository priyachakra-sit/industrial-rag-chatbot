# ============================================================
# INSIGHTFORGE AI
# FINAL VERSION — ChatGPT-style UI + RAG + Web Search
# ============================================================

import pandas as pd
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from tavily import TavilyClient
import tempfile
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="InsightForge AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# FULL UI — ChatGPT-style clean white/green design
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}

html, body, .stApp {
    background: #f0faf4 !important;
    color: #0d2e1c;
}

header, footer, #MainMenu,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="collapsedControl"] {
    display: none !important;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

[data-testid="stSidebar"] {
    display: none;
}

[data-testid="stChatMessage"] {
    background: white;
    border: 1px solid #e4f0ea;
    border-radius: 18px;
    padding: 18px 22px;
    margin-bottom: 14px;
    color: #0d2e1c;
    max-width: 820px;
    margin-left: auto;
    margin-right: auto;
    box-shadow: 0 2px 8px rgba(0,100,50,0.06);
}

[data-testid="stChatInput"] {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: #f0faf4;
    padding: 16px 20px 20px;
    border-top: 1px solid #d4ecdf;
    z-index: 999;
}

[data-testid="stChatInput"] > div {
    max-width: 780px;
    margin: 0 auto;
    background: white;
    border-radius: 50px;
    border: 1.5px solid #2d8653;
    box-shadow: 0 4px 20px rgba(45,134,83,0.15);
}

[data-testid="stChatInput"] textarea {
    font-size: 15px;
    color: #0d2e1c;
    padding: 14px 20px;
}

[data-testid="stFileUploader"] {
    max-width: 780px;
    margin: 0 auto 12px;
}

.chat-area {
    max-width: 860px;
    margin: 0 auto;
    padding: 76px 20px 160px;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    max-width: 800px;
    margin: 36px auto 0;
}

.feature-card {
    background: white;
    border: 1.5px solid #d4ecdf;
    border-radius: 18px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(45,134,83,0.07);
}

.feature-card .icon { font-size: 28px; margin-bottom: 10px; }
.feature-card .title { font-weight: 700; font-size: 14px; color: #0d2e1c; margin-bottom: 4px; }
.feature-card .subtitle { font-size: 12px; color: #5a8a6a; }

.hero {
    text-align: center;
    padding: 60px 20px 0;
}

.hero-icon {
    width: 80px;
    height: 80px;
    background: #e8f7ee;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 20px;
    font-size: 36px;
    border: 2px solid #c2e4cc;
}

.hero h1 {
    font-size: 44px;
    font-weight: 800;
    color: #0d5c30;
    margin: 0 0 8px;
    letter-spacing: -1px;
}

.hero p { font-size: 17px; color: #4a7a5a; margin: 0; }

.topbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 58px;
    background: rgba(240,250,244,0.96);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid #d4ecdf;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    z-index: 1000;
}

.topbar-logo {
    font-size: 20px;
    font-weight: 800;
    color: #0d5c30;
}

.file-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #e8f7ee;
    border: 1px solid #b8dfc8;
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 600;
    color: #0d5c30;
    margin: 4px;
}

.file-badges { text-align: center; padding: 8px 0; }

.ws-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #fff8e1;
    border: 1px solid #ffe082;
    border-radius: 12px;
    padding: 3px 10px;
    font-size: 11px;
    color: #795548;
    margin-bottom: 8px;
    font-weight: 500;
}

.disclaimer {
    text-align: center;
    font-size: 12px;
    color: #7aaa8a;
    padding: 6px 0 2px;
}

</style>
""", unsafe_allow_html=True)

# ── Top Nav ──
st.markdown("""
<div class="topbar">
    <span class="topbar-logo">⚡ InsightForge AI</span>
    <span style="font-size:13px;color:#4a7a5a;font-weight:500;">Your Intelligent Industrial Assistant</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# API CLIENTS
# ============================================================

groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are InsightForge AI — a smart, modern assistant like ChatGPT.

RULES:
- Be concise, natural, and helpful
- For greetings, reply briefly (1-2 lines max)
- Never ask unnecessary follow-up questions
- Only use structure/lists when it genuinely helps
- For document questions, answer using only the provided data
- For general questions, answer from your knowledge
- For web search results, summarise clearly and cite sources
- Never hallucinate data or make up numbers
- If data is unavailable, say so honestly
"""

# ============================================================
# WEB SEARCH
# ============================================================

def web_search(query):
    try:
        results = tavily.search(query=query, search_depth="advanced", max_results=3)
        web_context = ""
        sources = []
        for r in results["results"]:
            web_context += f"\nSOURCE: {r['title']}\nURL: {r['url']}\nCONTENT: {r['content']}\n"
            sources.append(r['url'])
        return web_context, sources
    except Exception:
        return "", []

# ============================================================
# DATA ANALYSIS
# ============================================================

def analyse_dataframe(df):
    analysis = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "missing_values": df.isnull().sum()[df.isnull().sum() > 0].to_dict()
    }
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        analysis["statistics"] = numeric_df.describe().round(2).to_dict()
    return analysis

# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# ============================================================
# FILE PROCESSING
# ============================================================

def process_files(uploaded_files):
    all_documents = []
    names = []

    for uploaded_file in uploaded_files:
        ext = uploaded_file.name.split('.')[-1].lower()
        names.append(uploaded_file.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        if ext == "pdf":
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            all_documents.extend(docs)

        elif ext == "xlsx":
            excel_data = pd.read_excel(temp_path, sheet_name=None)
            for sheet_name, df in excel_data.items():
                analysis = analyse_dataframe(df)
                text = f"SHEET: {sheet_name}\n\nDATA:\n{df.head(150).to_string(index=False)}\n\nANALYSIS:\n{analysis}"
                all_documents.append(Document(page_content=text))

        elif ext == "csv":
            df = pd.read_csv(temp_path)
            analysis = analyse_dataframe(df)
            text = f"FILE: {uploaded_file.name}\n\nDATA:\n{df.head(150).to_string(index=False)}\n\nANALYSIS:\n{analysis}"
            all_documents.append(Document(page_content=text))

        try:
            os.unlink(temp_path)
        except Exception:
            pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = splitter.split_documents(all_documents)
    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, names

# ============================================================
# WEB SEARCH DETECTION
# ============================================================

WEB_KEYWORDS = [
    "latest", "news", "today", "current", "recent", "2025", "2026",
    "trend", "update", "price", "stock", "weather", "who is", "what happened",
    "right now", "live", "breaking"
]

def needs_web_search(question):
    q = question.lower()
    return any(word in q for word in WEB_KEYWORDS)

# ============================================================
# LAYOUT
# ============================================================

st.markdown('<div class="chat-area">', unsafe_allow_html=True)

# Hero
if len(st.session_state.chat_history) == 0:
    st.markdown("""
    <div class="hero">
        <div class="hero-icon">💬</div>
        <h1>InsightForge AI</h1>
        <p>Your Intelligent Industrial Assistant</p>
    </div>

    <div class="feature-grid">
        <div class="feature-card">
            <div class="icon">📄</div>
            <div class="title">Upload File</div>
            <div class="subtitle">PDF, Excel, CSV</div>
        </div>
        <div class="feature-card">
            <div class="icon">🌐</div>
            <div class="title">Web Search</div>
            <div class="subtitle">Search the web</div>
        </div>
        <div class="feature-card">
            <div class="icon">📊</div>
            <div class="title">Data Analysis</div>
            <div class="subtitle">Analyse your data</div>
        </div>
        <div class="feature-card">
            <div class="icon">🧠</div>
            <div class="title">Get Insights</div>
            <div class="subtitle">Smart insights</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# File badges
if st.session_state.uploaded_file_names:
    badges = "".join(
        f'<span class="file-badge">📎 {name}</span>'
        for name in st.session_state.uploaded_file_names
    )
    st.markdown(f'<div class="file-badges">{badges}</div>', unsafe_allow_html=True)

# Chat history
for chat in st.session_state.chat_history[-10:]:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        if chat.get("web_searched"):
            st.markdown('<span class="ws-badge">🌐 Web Search Used</span>', unsafe_allow_html=True)
        st.write(chat["answer"])

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_files = st.file_uploader(
    "📎 Upload PDF, Excel, or CSV",
    type=["pdf", "xlsx", "csv"],
    accept_multiple_files=True,
    key="file_uploader"
)

if uploaded_files:
    with st.spinner("Processing your files..."):
        vectorstore, names = process_files(uploaded_files)
        st.session_state.vectorstore = vectorstore
        st.session_state.uploaded_file_names = names
    st.toast(f"✅ {len(names)} file(s) ready — ask anything about them!")
    st.rerun()

# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input("Ask me anything...")

# ============================================================
# HANDLE QUESTION
# ============================================================

if question:

    with st.chat_message("user"):
        st.write(question)

    web_searched = False
    sources = []

    # Decide mode
    if st.session_state.vectorstore is not None:
        # RAG
        docs = st.session_state.vectorstore.similarity_search(question, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""You are analysing uploaded documents/data.

DATA CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Answer using only the data provided above
- Be concise and specific
- Use tables or bullet points only if they help clarity
- If the answer is not in the data, say "This information is not in the uploaded files"
- Never hallucinate numbers or facts
"""

    elif needs_web_search(question):
        # Web search
        web_context, sources = web_search(question)
        web_searched = True
        prompt = f"""Answer the user's question using the latest web information below.

WEB SEARCH RESULTS:
{web_context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Summarise clearly from the search results
- Mention source URLs where helpful
- Be concise
"""

    else:
        # General chat
        prompt = question

    # Build messages with history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for chat in st.session_state.chat_history[-6:]:
        messages.append({"role": "user", "content": chat["question"]})
        messages.append({"role": "assistant", "content": chat["answer"]})
    messages.append({"role": "user", "content": prompt})

    # Stream response
    with st.chat_message("assistant"):
        if web_searched:
            st.markdown('<span class="ws-badge">🌐 Web Search Used</span>', unsafe_allow_html=True)

        placeholder = st.empty()
        full_response = ""

        try:
            stream = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.2,
                max_tokens=1500,
                stream=True
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    placeholder.markdown(full_response + "▌")

            if sources:
                src_text = "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources[:3])
                full_response += src_text

            placeholder.markdown(full_response)

        except Exception:
            placeholder.error("⚠️ Service temporarily busy. Please try again in a moment.")
            full_response = ""

    if full_response:
        st.session_state.chat_history.append({
            "question": question,
            "answer": full_response,
            "web_searched": web_searched
        })

# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("""
<div class="disclaimer">
    InsightForge AI can make mistakes. Verify important information. ⓘ
</div>
""", unsafe_allow_html=True)

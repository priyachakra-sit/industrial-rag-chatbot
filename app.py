# ============================================================
# INSIGHTFORGE AI — FINAL VERSION WITH NEW CHAT
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
# CSS
# ============================================================

st.markdown("""<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: #f7f9fc;
}

/* Main page */
.block-container{
    max-width:1000px !important;
    margin:auto !important;
    padding-top:20px !important;
    padding-bottom:100px !important;
}

/* Hide Streamlit junk */
#MainMenu,
footer,
header{
    visibility:hidden;
}

/* Chat messages */
[data-testid="stChatMessage"]{
    background:white !important;
    border:1px solid #e5e7eb !important;
    border-radius:18px !important;
    padding:18px !important;
    margin-bottom:14px !important;
    box-shadow:0 2px 8px rgba(0,0,0,.05);
}

/* Input */
[data-testid="stChatInput"]{
    margin-top:20px;
}

[data-testid="stChatInput"] > div{
    border-radius:20px !important;
    border:1px solid #d1d5db !important;
    background:white !important;
    box-shadow:0 4px 15px rgba(0,0,0,.08);
}

/* Buttons */
.stButton button{
    border-radius:12px !important;
    background:#1f7a4d !important;
    color:white !important;
    border:none !important;
}

/* File uploader */
[data-testid="stFileUploader"]{
    background:white;
    border-radius:15px;
    padding:10px;
    border:1px dashed #cbd5e1;
}

</style>""", unsafe_allow_html=True)


# ============================================================
# CLIENTS
# ============================================================

groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

# ============================================================
# SESSION STATE
# ============================================================

for k, v in {
    "chat_history": [],
    "vectorstore": None,
    "uploaded_file_names": [],
    "show_uploader": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are InsightForge AI — a smart, modern AI assistant like ChatGPT.
- Be concise, natural, and helpful
- Keep greetings very short (1-2 lines)
- Never ask unnecessary follow-up questions
- Use bullet points only when they genuinely help
- For document questions: answer only from provided data, never hallucinate
- For general questions: answer from your knowledge
- For web results: summarise clearly and cite sources
- Say "This information is not in the uploaded files" if data is missing
"""

# ============================================================
# HELPERS
# ============================================================

def web_search(query):
    try:
        res = tavily.search(query=query, search_depth="advanced", max_results=3)
        ctx, srcs = "", []
        for r in res["results"]:
            ctx += f"\nSOURCE: {r['title']}\nURL: {r['url']}\nCONTENT: {r['content']}\n"
            srcs.append(r['url'])
        return ctx, srcs
    except Exception:
        return "", []

def analyse_df(df):
    info = {
        "rows": df.shape[0], "columns": df.shape[1],
        "column_names": list(df.columns),
        "missing": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
    }
    num = df.select_dtypes(include='number')
    if not num.empty:
        info["stats"] = num.describe().round(2).to_dict()
    return info

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def process_files(files):
    all_docs, names = [], []
    for f in files:
        ext = f.name.split('.')[-1].lower()
        names.append(f.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(f.read())
            path = tmp.name
        if ext == "pdf":
            all_docs.extend(PyPDFLoader(path).load())
        elif ext == "xlsx":
            for sname, df in pd.read_excel(path, sheet_name=None).items():
                text = f"SHEET: {sname}\n\nDATA:\n{df.head(150).to_string(index=False)}\n\nSTATS:\n{analyse_df(df)}"
                all_docs.append(Document(page_content=text))
        elif ext == "csv":
            df = pd.read_csv(path)
            text = f"FILE: {f.name}\n\nDATA:\n{df.head(150).to_string(index=False)}\n\nSTATS:\n{analyse_df(df)}"
            all_docs.append(Document(page_content=text))
        try: os.unlink(path)
        except: pass
    chunks = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250).split_documents(all_docs)
    return FAISS.from_documents(chunks, load_embeddings()), names

WEB_KEYWORDS = ["latest","news","today","current","recent","2025","2026",
                 "trend","update","price","stock","weather","who is",
                 "what happened","right now","live","breaking"]

def needs_web(q):
    return any(w in q.lower() for w in WEB_KEYWORDS)

# ============================================================
# TOP NAVBAR (HTML — logo + subtitle only)
# ============================================================


# ── NEW CHAT BUTTON — injected via a fixed-position div ──
col1, col2 = st.columns([8,1])

with col1:
    st.title("⚡ InsightForge AI")

with col2:
    if st.button("New Chat"):
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.session_state.uploaded_file_names = []
        st.rerun()
    
# ── DISCLAIMER ──
st.markdown("""
<div class="disclaimer-bar">
    InsightForge AI can make mistakes. Verify important information. ⓘ
</div>
""", unsafe_allow_html=True)

# ============================================================
# HERO — only when no chat history
# ============================================================


# ============================================================
# FILE BADGES
# ============================================================

if st.session_state.uploaded_file_names:

    st.info(
        "📎 Uploaded Files: " +
        ", ".join(st.session_state.uploaded_file_names)
    )
    badges = "".join(f'<span class="file-badge">📎 {n}</span>' for n in st.session_state.uploaded_file_names)
    st.markdown(f'<div style="text-align:center;margin:10px 0 16px;">{badges}</div>', unsafe_allow_html=True)

# ============================================================
# CHAT HISTORY
# ============================================================

for chat in st.session_state.chat_history[-12:]:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        if chat.get("web_searched"):
            st.markdown('<span class="ws-badge">🌐 Web Search Used</span>', unsafe_allow_html=True)
        st.write(chat["answer"])

# ============================================================
# ATTACH FILE BUTTON + UPLOADER
# ============================================================


# ============================================================
# CHAT INPUT
# ============================================================
# SHOW UPLOADER ONLY ONCE

if len(st.session_state.uploaded_file_names) == 0:

    uploaded_files = st.file_uploader(
        "📎 Upload PDF, Excel or CSV",
        accept_multiple_files=True,
        type=["pdf", "xlsx", "csv"]
    )

    if uploaded_files:

        vectorstore, names = process_files(uploaded_files)

        st.session_state.vectorstore = vectorstore
        st.session_state.uploaded_file_names = names

        st.success("✅ Files uploaded successfully")

        st.rerun()
question = st.chat_input("Ask me anything...")

# ============================================================
# HANDLE QUESTION
# ============================================================

if question:

    with st.chat_message("user"):
        st.write(question)

    web_searched = False
    sources = []

    if st.session_state.vectorstore is not None:
        docs = st.session_state.vectorstore.similarity_search(question, k=5)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"""Analyse the uploaded data and answer the question.

DATA CONTEXT:
{context}

QUESTION: {question}

Rules: Use only the data above. Be concise. Never hallucinate. If not found, say so."""

    elif needs_web(question):
        web_ctx, sources = web_search(question)
        web_searched = True
        prompt = f"""Answer using the web search results below.

WEB RESULTS:
{web_ctx}

QUESTION: {question}

Summarise clearly, cite sources, be concise."""

    else:
        prompt = question

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for c in st.session_state.chat_history[-6:]:
        messages.append({"role": "user", "content": c["question"]})
        messages.append({"role": "assistant", "content": c["answer"]})
    messages.append({"role": "user", "content": prompt})

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
                full_response += "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources[:3])
            placeholder.markdown(full_response)

        except Exception:
            placeholder.error("⚠️ Service temporarily busy. Please try again.")
            full_response = ""

    if full_response:
        st.session_state.chat_history.append({
            "question": question,
            "answer": full_response,
            "web_searched": web_searched
        })
        #st.rerun()

# ============================================================
# INSIGHTFORGE AI — CLEAN FINAL VERSION
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

html, body, .stApp { background: #f0faf4 !important; color: #0d2e1c !important; }

header, footer, #MainMenu,
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="collapsedControl"],
[data-testid="stStatusWidget"], .stDeployButton {
    display: none !important;
}

.block-container { padding: 70px 0 160px !important; max-width: 100% !important; }

[data-testid="stSidebar"] { display: none !important; }

/* TOP NAV */
.topbar {
    position: fixed; top: 0; left: 0; right: 0;
    height: 60px;
    background: rgba(240,250,244,0.97);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid #d4ecdf;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 32px;
    z-index: 9999;
}
.topbar-logo { font-size: 21px; font-weight: 800; color: #0d5c30; }
.topbar-sub { font-size: 13px; color: #5a8a6a; }

/* CHAT MESSAGES */
[data-testid="stChatMessage"] {
    background: white !important;
    border: 1px solid #e0ede6 !important;
    border-radius: 18px !important;
    padding: 16px 22px !important;
    margin-bottom: 12px !important;
    max-width: 780px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    box-shadow: 0 2px 8px rgba(0,100,50,0.05) !important;
}

/* CHAT INPUT */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 40px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 780px !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    z-index: 9998 !important;
}

[data-testid="stChatInput"] > div {
    background: white !important;
    border: 2px solid #2d8653 !important;
    border-radius: 50px !important;
    box-shadow: 0 4px 24px rgba(45,134,83,0.15) !important;
    min-height: 58px !important;
}

[data-testid="stChatInput"] textarea {
    font-size: 15px !important;
    padding: 16px 22px !important;
    color: #0d2e1c !important;
}

[data-testid="stChatInput"] button {
    background: #2d8653 !important;
    border-radius: 50% !important;
    margin: 8px !important;
    width: 42px !important; height: 42px !important;
}

/* DISCLAIMER */
.disclaimer-bar {
    position: fixed; bottom: 10px; left: 0; right: 0;
    text-align: center;
    font-size: 11px; color: #8aaa96;
    z-index: 9997;
}

/* WEB BADGE */
.ws-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: #fff8e1; border: 1px solid #ffe082;
    border-radius: 12px; padding: 3px 10px;
    font-size: 11px; color: #795548; margin-bottom: 8px; font-weight: 500;
}

/* FILE BADGE */
.file-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #e8f7ee; border: 1px solid #b8dfc8;
    border-radius: 20px; padding: 4px 12px;
    font-size: 12px; font-weight: 600; color: #0d5c30; margin: 3px;
}

/* HERO SECTION */
.hero-wrap {
    text-align: center;
    padding: 50px 20px 30px;
    max-width: 860px;
    margin: 0 auto;
}

.hero-bubble {
    width: 90px; height: 90px;
    background: #e2f4ea;
    border-radius: 50%;
    display: inline-flex;
    align-items: center; justify-content: center;
    font-size: 42px;
    border: 2px solid #c2dece;
    margin-bottom: 20px;
}

.hero-title {
    font-size: 44px; font-weight: 800;
    color: #0d5c30; margin: 0 0 8px;
    letter-spacing: -1px;
}

.hero-sub { font-size: 16px; color: #4a7a5a; margin: 0 0 36px; }

/* FEATURE CARDS */
.cards-row {
    display: flex; gap: 14px; justify-content: center;
    margin-bottom: 20px;
}

.fcard {
    background: white;
    border: 1.5px solid #d4ecdf;
    border-radius: 18px;
    padding: 22px 18px 18px;
    width: 168px; text-align: center;
    box-shadow: 0 2px 12px rgba(45,134,83,0.07);
    flex-shrink: 0;
}

.fcard .fci { font-size: 28px; margin-bottom: 10px; }
.fcard .fct { font-weight: 700; font-size: 13px; color: #0d2e1c; margin-bottom: 4px; }
.fcard .fcs { font-size: 11px; color: #5a8a6a; }

/* ATTACH BTN */
.stButton > button {
    background: white !important;
    border: 1.5px solid #c2dece !important;
    border-radius: 30px !important;
    color: #0d5c30 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 6px 16px !important;
    height: auto !important;
    width: auto !important;
}

.stButton > button:hover {
    border-color: #2d8653 !important;
    background: #e8f7ee !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    max-width: 780px;
    margin: 0 auto 10px;
    background: white;
    border: 1.5px dashed #b8dfc8;
    border-radius: 14px;
    padding: 10px 16px;
}

</style>
""", unsafe_allow_html=True)

# ── TOP NAVBAR ──
st.markdown("""
<div class="topbar">
    <span class="topbar-sub">Your Intelligent Industrial Assistant</span>
</div>
""", unsafe_allow_html=True)

# ── DISCLAIMER ──
st.markdown("""
<div class="disclaimer-bar">
    InsightForge AI can make mistakes. Verify important information. ⓘ
</div>
""", unsafe_allow_html=True)

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
# HERO — only when no chat history
# ============================================================

if len(st.session_state.chat_history) == 0:
    st.markdown("""
<div class="hero-wrap">
    <div class="hero-bubble">💬</div>
    <div class="hero-title">InsightForge AI</div>
    <div class="hero-sub">Your Intelligent Industrial Assistant</div>
    <div class="cards-row">
        <div class="fcard"><div class="fci">📄</div><div class="fct">Upload File</div><div class="fcs">PDF, Excel, CSV</div></div>
        <div class="fcard"><div class="fci">🌐</div><div class="fct">Web Search</div><div class="fcs">Search the web</div></div>
        <div class="fcard"><div class="fci">📊</div><div class="fct">Data Analysis</div><div class="fcs">Analyse your data</div></div>
        <div class="fcard"><div class="fci">🧠</div><div class="fct">Get Insights</div><div class="fcs">Smart insights</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# FILE BADGES
# ============================================================

if st.session_state.uploaded_file_names:
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
# ATTACH FILE BUTTON (above chat input)
# ============================================================

left, mid, right = st.columns([2, 6, 2])
with left:
    if st.button("📎  Attach File"):
        st.session_state.show_uploader = not st.session_state.show_uploader

# ============================================================
# FILE UPLOADER (shown only when toggled)
# ============================================================

if st.session_state.show_uploader:
    uploaded_files = st.file_uploader(
        "Upload PDF, Excel or CSV",
        type=["pdf", "xlsx", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded_files:
        with st.spinner("Processing files..."):
            vs, names = process_files(uploaded_files)
            st.session_state.vectorstore = vs
            st.session_state.uploaded_file_names = names
            st.session_state.show_uploader = False
        st.toast(f"✅ {len(names)} file(s) ready!")
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
        st.rerun()

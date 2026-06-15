# ============================================================
# INSIGHTFORGE AI — FINAL CLEAN VERSION
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

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after {
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}

html, body, .stApp {
    background: #f0faf4 !important;
    color: #0d2e1c !important;
    margin: 0;
    padding: 0;
}

/* ── Hide ALL Streamlit chrome ── */
header, footer, #MainMenu,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="collapsedControl"],
[data-testid="stStatusWidget"],
.stDeployButton {
    display: none !important;
    visibility: hidden !important;
}

/* ── Remove default block padding ── */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
    margin: 0 !important;
}

/* ── Sidebar gone ── */
[data-testid="stSidebar"] { display: none !important; }

/* ── TOP NAVBAR ── */
.topbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 60px;
    background: rgba(240,250,244,0.97);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid #d4ecdf;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 32px;
    z-index: 9999;
}

.topbar-logo {
    font-size: 21px;
    font-weight: 800;
    color: #0d5c30;
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: -0.5px;
}

.topbar-sub {
    font-size: 13px;
    color: #5a8a6a;
    font-weight: 400;
}

/* ── SCROLLABLE CHAT AREA ── */
.chat-scroll {
    position: fixed;
    top: 60px;
    left: 0; right: 0;
    bottom: 110px;
    overflow-y: auto;
    padding: 40px 0 20px;
}

.chat-inner {
    max-width: 760px;
    margin: 0 auto;
    padding: 0 20px;
}

/* ── HERO ── */
.hero {
    text-align: center;
    padding: 60px 20px 40px;
}

.hero-bubble {
    width: 88px;
    height: 88px;
    background: #e2f4ea;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 40px;
    margin-bottom: 22px;
    border: 2px solid #c2dece;
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    color: #0d5c30;
    margin: 0 0 8px;
    letter-spacing: -1px;
}

.hero p {
    font-size: 16px;
    color: #4a7a5a;
    margin: 0 0 40px;
}

/* ── FEATURE CARDS ── */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 40px;
}

.feature-card {
    background: white;
    border: 1.5px solid #d4ecdf;
    border-radius: 18px;
    padding: 22px 14px 18px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(45,134,83,0.07);
    transition: all 0.2s;
}

.feature-card:hover {
    border-color: #2d8653;
    box-shadow: 0 6px 20px rgba(45,134,83,0.14);
    transform: translateY(-2px);
}

.feature-card .fc-icon { font-size: 28px; margin-bottom: 10px; }
.feature-card .fc-title { font-weight: 700; font-size: 13px; color: #0d2e1c; margin-bottom: 4px; }
.feature-card .fc-sub { font-size: 11px; color: #5a8a6a; }

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: white !important;
    border: 1px solid #e4f0ea !important;
    border-radius: 18px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    color: #0d2e1c !important;
    box-shadow: 0 2px 8px rgba(0,100,50,0.05) !important;
}

/* ── BOTTOM BAR (input + upload + disclaimer) ── */
.bottom-bar {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: #f0faf4;
    border-top: 1px solid #d4ecdf;
    padding: 12px 20px 14px;
    z-index: 9998;
}

.bottom-inner {
    max-width: 760px;
    margin: 0 auto;
}

/* ── Chat input styling ── */
[data-testid="stChatInput"] {
    position: static !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] > div {
    background: white !important;
    border: 2px solid #2d8653 !important;
    border-radius: 50px !important;
    box-shadow: 0 4px 20px rgba(45,134,83,0.12) !important;
    min-height: 56px !important;
}

[data-testid="stChatInput"] textarea {
    font-size: 15px !important;
    padding: 14px 20px !important;
    color: #0d2e1c !important;
    min-height: 54px !important;
    line-height: 1.5 !important;
}

[data-testid="stChatInput"] button {
    background: #2d8653 !important;
    border-radius: 50% !important;
    margin: 6px !important;
    width: 40px !important;
    height: 40px !important;
}

/* ── Upload button row ── */
.upload-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}

.upload-trigger-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: white;
    border: 1.5px solid #c2dece;
    border-radius: 30px;
    padding: 7px 16px;
    font-size: 13px;
    font-weight: 600;
    color: #0d5c30;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
}

.upload-trigger-btn:hover {
    border-color: #2d8653;
    background: #e8f7ee;
}

/* ── File uploader — hide when not in upload mode ── */
.file-uploader-wrap [data-testid="stFileUploader"] {
    background: white;
    border: 1.5px dashed #b8dfc8;
    border-radius: 14px;
    padding: 10px 16px;
    margin-bottom: 8px;
}

.file-uploader-wrap [data-testid="stFileUploaderDropzoneInstructions"] {
    display: none;
}

/* ── File badges ── */
.file-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #e8f7ee;
    border: 1px solid #b8dfc8;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    color: #0d5c30;
    margin-right: 6px;
}

/* ── Web badge ── */
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

/* ── Disclaimer ── */
.disclaimer {
    text-align: center;
    font-size: 11px;
    color: #8aaa96;
    margin-top: 8px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# CLIENTS
# ============================================================

groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "chat_history": [],
    "vectorstore": None,
    "uploaded_file_names": [],
    "show_uploader": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are InsightForge AI — a smart, modern assistant like ChatGPT.

RULES:
- Be concise, natural, and helpful
- For greetings, reply briefly (1-2 lines max)
- Never ask unnecessary follow-up questions
- Only use structure/lists when it genuinely helps clarity
- For document questions, answer using only the provided data
- For general questions, answer from your knowledge
- For web search results, summarise clearly and cite sources
- Never hallucinate data or make up numbers
- If data is unavailable in documents, say so honestly
"""

# ============================================================
# HELPERS
# ============================================================

def web_search(query):
    try:
        results = tavily.search(query=query, search_depth="advanced", max_results=3)
        ctx = ""
        srcs = []
        for r in results["results"]:
            ctx += f"\nSOURCE: {r['title']}\nURL: {r['url']}\nCONTENT: {r['content']}\n"
            srcs.append(r['url'])
        return ctx, srcs
    except Exception:
        return "", []


def analyse_df(df):
    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "missing": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
    }
    num = df.select_dtypes(include='number')
    if not num.empty:
        info["statistics"] = num.describe().round(2).to_dict()
    return info


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def process_files(files):
    docs = []
    names = []
    for f in files:
        ext = f.name.split('.')[-1].lower()
        names.append(f.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(f.read())
            path = tmp.name

        if ext == "pdf":
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        elif ext == "xlsx":
            sheets = pd.read_excel(path, sheet_name=None)
            for sname, df in sheets.items():
                info = analyse_df(df)
                text = f"SHEET: {sname}\n\nDATA:\n{df.head(150).to_string(index=False)}\n\nSTATS:\n{info}"
                docs.append(Document(page_content=text))

        elif ext == "csv":
            df = pd.read_csv(path)
            info = analyse_df(df)
            text = f"FILE: {f.name}\n\nDATA:\n{df.head(150).to_string(index=False)}\n\nSTATS:\n{info}"
            docs.append(Document(page_content=text))

        try:
            os.unlink(path)
        except Exception:
            pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = splitter.split_documents(docs)
    emb = load_embeddings()
    vs = FAISS.from_documents(chunks, emb)
    return vs, names


WEB_KEYWORDS = [
    "latest", "news", "today", "current", "recent", "2025", "2026",
    "trend", "update", "price", "stock", "weather", "who is",
    "what happened", "right now", "live", "breaking"
]

def needs_web(q):
    return any(w in q.lower() for w in WEB_KEYWORDS)

# ============================================================
# TOP NAVBAR
# ============================================================

st.markdown("""
<div class="topbar">
    <div class="topbar-logo">⚡ InsightForge AI</div>
    <div class="topbar-sub">Your Intelligent Industrial Assistant</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SCROLLABLE CHAT AREA
# ============================================================

st.markdown('<div class="chat-scroll"><div class="chat-inner">', unsafe_allow_html=True)

# ── Hero — only when no chat ──
if len(st.session_state.chat_history) == 0:
    st.markdown("""
    <div class="hero">
        <div class="hero-bubble">💬</div>
        <h1>InsightForge AI</h1>
        <p>Your Intelligent Industrial Assistant</p>

        <div class="feature-grid">
            <div class="feature-card">
                <div class="fc-icon">📄</div>
                <div class="fc-title">Upload File</div>
                <div class="fc-sub">PDF, Excel, CSV</div>
            </div>
            <div class="feature-card">
                <div class="fc-icon">🌐</div>
                <div class="fc-title">Web Search</div>
                <div class="fc-sub">Search the web</div>
            </div>
            <div class="feature-card">
                <div class="fc-icon">📊</div>
                <div class="fc-title">Data Analysis</div>
                <div class="fc-sub">Analyse your data</div>
            </div>
            <div class="feature-card">
                <div class="fc-icon">🧠</div>
                <div class="fc-title">Get Insights</div>
                <div class="fc-sub">Smart insights</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── File badges ──
if st.session_state.uploaded_file_names:
    badges = "".join(f'<span class="file-badge">📎 {n}</span>' for n in st.session_state.uploaded_file_names)
    st.markdown(f'<div style="text-align:center;margin-bottom:16px;">{badges}</div>', unsafe_allow_html=True)

# ── Chat messages ──
for chat in st.session_state.chat_history[-12:]:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        if chat.get("web_searched"):
            st.markdown('<span class="ws-badge">🌐 Web Search Used</span>', unsafe_allow_html=True)
        st.write(chat["answer"])

st.markdown('</div></div>', unsafe_allow_html=True)

# ============================================================
# BOTTOM BAR
# ============================================================

st.markdown('<div class="bottom-bar"><div class="bottom-inner">', unsafe_allow_html=True)

# ── Upload toggle button + file badges inline ──
col_btn, col_badges = st.columns([1, 4])

with col_btn:
    if st.button("📎  Attach File", key="toggle_upload", use_container_width=False):
        st.session_state.show_uploader = not st.session_state.show_uploader

# ── File uploader — only visible when toggled ──
if st.session_state.show_uploader:
    st.markdown('<div class="file-uploader-wrap">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDF, Excel or CSV",
        type=["pdf", "xlsx", "csv"],
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        with st.spinner("Processing files..."):
            vs, names = process_files(uploaded_files)
            st.session_state.vectorstore = vs
            st.session_state.uploaded_file_names = names
            st.session_state.show_uploader = False
        st.toast(f"✅ {len(names)} file(s) ready!")
        st.rerun()
else:
    uploaded_files = None

# ── Chat input ──
question = st.chat_input("Ask me anything...")

# ── Disclaimer ──
st.markdown('<div class="disclaimer">InsightForge AI can make mistakes. Verify important information. ⓘ</div>', unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)

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
        docs = st.session_state.vectorstore.similarity_search(question, k=5)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"""You are analysing uploaded documents/data.

DATA CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Answer using only the data provided
- Be concise and specific
- Use tables or bullet points only when they genuinely help
- If the answer is not in the data, say "This information is not in the uploaded files"
- Never hallucinate numbers or facts
"""

    elif needs_web(question):
        web_context, sources = web_search(question)
        web_searched = True
        prompt = f"""Answer using the latest web information below.

WEB SEARCH RESULTS:
{web_context}

USER QUESTION:
{question}

- Summarise clearly, mention sources where helpful, be concise
"""

    else:
        prompt = question

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for chat in st.session_state.chat_history[-6:]:
        messages.append({"role": "user", "content": chat["question"]})
        messages.append({"role": "assistant", "content": chat["answer"]})
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
        st.rerun()

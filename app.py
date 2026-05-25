# ============================================================
# FUTURISTIC GLASSMORPHISM INDUSTRIAL AI ASSISTANT
# COMPLETE PREMIUM CHATGPT STYLE UI
# ============================================================

import pandas as pd
import plotly.express as px
from langchain_core.documents import Document
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import tempfile

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Industrial AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ============================================================
BACKGROUND
============================================================ */

.stApp {

    background:
        radial-gradient(circle at top left, rgba(168,85,247,0.18), transparent 25%),
        radial-gradient(circle at top right, rgba(59,130,246,0.18), transparent 25%),
        linear-gradient(180deg, #050816 0%, #0B1023 100%);

    color: white;
}

/* ============================================================
REMOVE STREAMLIT DEFAULT
============================================================ */

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

[data-testid="stHeader"] {
    display: none;
}

[data-testid="stToolbar"] {
    display: none;
}

/* ============================================================
MAIN CONTAINER
============================================================ */

.block-container {

    padding-top: 1rem !important;

    padding-left: 2rem;

    padding-right: 2rem;

    max-width: 1500px;
}

/* ============================================================
SIDEBAR
============================================================ */

[data-testid="stSidebar"] {

    background:
        rgba(15,23,42,0.72);

    backdrop-filter:
        blur(20px);

    border-right:
        1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* ============================================================
SIDEBAR LOGO
============================================================ */

.sidebar-logo {

    font-size: 28px;

    font-weight: 800;

    margin-bottom: 40px;

    background:
        linear-gradient(90deg,#A855F7,#38BDF8);

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

/* ============================================================
MENU ITEMS
============================================================ */

.menu-item {

    padding: 14px 18px;

    border-radius: 14px;

    margin-bottom: 12px;

    font-size: 15px;

    font-weight: 500;

    transition: 0.3s;

    cursor: pointer;

    background:
        rgba(255,255,255,0.03);

    border:
        1px solid rgba(255,255,255,0.04);
}

.menu-item:hover {

    transform:
        translateX(5px);

    background:
        rgba(168,85,247,0.15);

    border:
        1px solid rgba(168,85,247,0.25);
}

/* ============================================================
HERO SECTION
============================================================ */

.hero-title {

    text-align: center;

    font-size: 4rem;

    font-weight: 800;

    margin-top: 10px;

    background:
        linear-gradient(90deg,#A855F7,#38BDF8);

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.hero-subtitle {

    text-align: center;

    color: #CBD5E1;

    margin-bottom: 40px;

    font-size: 1.1rem;
}

/* ============================================================
SEARCH BAR
============================================================ */

.stTextInput > div > div > input {

    background:
        rgba(255,255,255,0.05);

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius:
        20px;

    height:
        55px;

    color:
        white !important;

    backdrop-filter:
        blur(12px);

    box-shadow:
        0px 10px 30px rgba(0,0,0,0.25);
}

/* ============================================================
MODE BOX
============================================================ */

.mode-box {

    background:
        rgba(255,255,255,0.04);

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius:
        22px;

    padding:
        20px;

    backdrop-filter:
        blur(18px);

    margin-bottom:
        30px;

    box-shadow:
        0px 10px 30px rgba(0,0,0,0.2);
}

/* ============================================================
KPI CARDS
============================================================ */

.kpi-card {

    background:
        rgba(255,255,255,0.05);

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius:
        24px;

    padding:
        24px;

    backdrop-filter:
        blur(18px);

    min-height:
        160px;

    transition:
        0.35s;

    box-shadow:
        0px 10px 30px rgba(0,0,0,0.25);
}

.kpi-card:hover {

    transform:
        translateY(-6px);

    border:
        1px solid rgba(168,85,247,0.35);

    box-shadow:
        0px 20px 40px rgba(168,85,247,0.18);
}

.kpi-title {

    color:
        #CBD5E1;

    font-size:
        13px;

    font-weight:
        700;

    margin-bottom:
        14px;
}

.kpi-value {

    font-size:
        2.2rem;

    font-weight:
        800;

    color:
        white;
}

/* ============================================================
CHAT MESSAGES
============================================================ */

[data-testid="stChatMessage"] {

    background:
        rgba(255,255,255,0.05);

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius:
        22px;

    padding:
        18px;

    margin-bottom:
        16px;

    backdrop-filter:
        blur(18px);

    box-shadow:
        0px 8px 25px rgba(0,0,0,0.22);

    animation:
        fadeIn 0.3s ease;
}

/* ============================================================
CHAT INPUT
============================================================ */

.stChatInput {

    position:
        sticky;

    bottom:
        8px;

    padding-top:
        12px;
}

.stChatInput > div {

    background:
        rgba(15,23,42,0.85);

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius:
        24px;

    backdrop-filter:
        blur(18px);

    padding:
        8px;

    box-shadow:
        0px 12px 30px rgba(0,0,0,0.28);
}

.stChatInput input {

    background:
        transparent !important;

    color:
        white !important;
}

/* ============================================================
BUTTONS
============================================================ */

.stButton button {

    width:
        100%;

    border-radius:
        18px;

    border:
        none;

    background:
        linear-gradient(135deg,#7C3AED,#4F46E5);

    color:
        white !important;

    font-weight:
        600;

    height:
        50px;

    transition:
        0.3s;

    box-shadow:
        0px 10px 25px rgba(124,58,237,0.28);
}

.stButton button:hover {

    transform:
        translateY(-3px);

    box-shadow:
        0px 16px 35px rgba(124,58,237,0.4);
}

/* ============================================================
FILE UPLOADER
============================================================ */

[data-testid="stFileUploader"] {

    background:
        rgba(255,255,255,0.05);

    border:
        2px dashed rgba(168,85,247,0.45);

    border-radius:
        22px;

    padding:
        30px;

    backdrop-filter:
        blur(18px);
}

/* ============================================================
RADIO BUTTON
============================================================ */

.stRadio > div {

    background:
        rgba(255,255,255,0.04);

    border:
        1px solid rgba(255,255,255,0.06);

    padding:
        15px;

    border-radius:
        18px;
}

/* ============================================================
ANIMATION
============================================================ */

@keyframes fadeIn {

    from {
        opacity: 0;
        transform: translateY(10px);
    }

    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

/* ============================================================
SCROLLBAR
============================================================ */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {

    background:
        linear-gradient(180deg,#7C3AED,#38BDF8);

    border-radius:
        10px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "files_processed" not in st.session_state:
    st.session_state.files_processed = False

if "report_count" not in st.session_state:
    st.session_state.report_count = 0

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "🧠 General AI Chat"

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">
    ⚡ Industrial AI
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="menu-item">🏠 Dashboard</div>', unsafe_allow_html=True)

    st.markdown('<div class="menu-item">📂 Reports</div>', unsafe_allow_html=True)

    st.markdown('<div class="menu-item">🤖 AI Analytics</div>', unsafe_allow_html=True)

    st.markdown('<div class="menu-item">📊 Insights</div>', unsafe_allow_html=True)

    st.markdown('<div class="menu-item">⚙️ Settings</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("➕ New Analysis"):

        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.session_state.files_processed = False
        st.session_state.report_count = 0
        st.session_state.uploaded_files = []

        st.rerun()

# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero-title">
⚡ Industrial AI Assistant
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-subtitle">
AI-powered Industrial Analytics & RAG Platform
</div>
""", unsafe_allow_html=True)

# ============================================================
# SEARCH BAR
# ============================================================

st.text_input(
    "",
    placeholder="Search reports, analytics, AI insights..."
)

# ============================================================
# MODE SECTION
# ============================================================

st.markdown('<div class="mode-box">', unsafe_allow_html=True)

chat_mode = st.radio(
    "Select AI Mode",
    [
        "🧠 General AI Chat",
        "📂 Upload & Analyze Reports"
    ],
    horizontal=True
)

st.session_state.chat_mode = chat_mode

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# KPI CARDS
# ============================================================

report_count = st.session_state.report_count
query_count = len(st.session_state.chat_history)

efficiency = 100 if query_count == 0 else min(100, 95 + query_count)

status = "ACTIVE" if st.session_state.files_processed else "WAITING"

col1, col2, col3, col4 = st.columns(4)

cards = [
    ("📄 REPORTS", report_count),
    ("🤖 AI QUERIES", query_count),
    ("⚡ EFFICIENCY", f"{efficiency}%"),
    ("🛰 STATUS", status)
]

for col, card in zip([col1,col2,col3,col4], cards):

    with col:

        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{card[0]}</div>
            <div class="kpi-value">{card[1]}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# FILE UPLOADER
# ============================================================

if st.session_state.chat_mode == "📂 Upload & Analyze Reports":

    st.markdown("## 📂 Upload Industrial Reports")

    uploaded_files = st.file_uploader(
        "Upload Reports",
        type=["pdf", "xlsx", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.session_state.uploaded_files = uploaded_files

        try:

            st.session_state.report_count = len(uploaded_files)

            all_documents = []

            for uploaded_file in uploaded_files:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f".{uploaded_file.name.split('.')[-1]}"
                ) as tmp:

                    tmp.write(uploaded_file.read())

                    temp_path = tmp.name

                if uploaded_file.name.endswith(".pdf"):

                    loader = PyPDFLoader(temp_path)

                    documents = loader.load()

                    all_documents.extend(documents)

                elif uploaded_file.name.endswith(".xlsx"):

                    excel_data = pd.read_excel(
                        temp_path,
                        sheet_name=None
                    )

                    text = ""

                    for sheet_name, df in excel_data.items():

                        text += f"\n\nSheet: {sheet_name}\n"

                        text += df.to_string(index=False)

                    all_documents.append(
                        Document(page_content=text)
                    )

                elif uploaded_file.name.endswith(".csv"):

                    df = pd.read_csv(temp_path)

                    text = df.to_string(index=False)

                    all_documents.append(
                        Document(page_content=text)
                    )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=150
            )

            chunks = splitter.split_documents(all_documents)

            @st.cache_resource
            def load_embeddings():

                return HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-en-v1.5"
                )

            embeddings = load_embeddings()

            st.session_state.vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

            st.session_state.files_processed = True

            st.success(
                f"✅ {len(uploaded_files)} file(s) processed successfully!"
            )

        except Exception as e:

            st.error(f"Error processing file: {e}")

# ============================================================
# DISPLAY CHAT
# ============================================================

for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])

# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask your Industrial AI question..."
)

# ============================================================
# GENERAL AI CHAT
# ============================================================

if (
    question
    and st.session_state.chat_mode
    == "🧠 General AI Chat"
):

    with st.chat_message("user"):
        st.write(question)

    api_key = st.secrets["GROQ_API_KEY"]

    client = Groq(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": """
You are a futuristic industrial AI assistant.

Be modern, intelligent, conversational and professional.
"""
        }
    ]

    for chat in st.session_state.chat_history[-5:]:

        messages.append(
            {
                "role": "user",
                "content": chat["question"]
            }
        )

        messages.append(
            {
                "role": "assistant",
                "content": chat["answer"]
            }
        )

    messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=1500,
            stream=True
        )

        for chunk in response:

            delta = chunk.choices[0].delta.content

            if delta:

                full_response += delta

                placeholder.markdown(
                    full_response + "▌"
                )

        placeholder.markdown(full_response)

    st.session_state.chat_history.append({
        "question": question,
        "answer": full_response
    })

# ============================================================
# RAG AI RESPONSE
# ============================================================

if (
    question
    and st.session_state.vectorstore is not None
    and st.session_state.chat_mode
    == "📂 Upload & Analyze Reports"
):

    with st.chat_message("user"):
        st.write(question)

    docs = st.session_state.vectorstore.similarity_search(
        question,
        k=5
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are an advanced industrial AI analyst assistant.

Rules:
- Be conversational
- Give insights
- Explain clearly
- Give recommendations
- Answer ONLY from context

CONTEXT:
{context}

QUESTION:
{question}
"""

    api_key = st.secrets["GROQ_API_KEY"]

    client = Groq(api_key=api_key)

    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1800,
            stream=True
        )

        for chunk in response:

            delta = chunk.choices[0].delta.content

            if delta:

                full_response += delta

                placeholder.markdown(
                    full_response + "▌"
                )

        placeholder.markdown(full_response)

    st.session_state.chat_history.append({
        "question": question,
        "answer": full_response
    })
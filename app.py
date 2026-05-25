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

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Industrial AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* =====================================================
MAIN APP
===================================================== */
.stApp {
    background: linear-gradient(135deg, #f5f7fb 0%, #eef2ff 100%);
}

/* =====================================================
REMOVE STREAMLIT DEFAULT
===================================================== */
header {
    visibility: hidden;
    height: 0px;
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

[data-testid="stDecoration"] {
    display: none;
}

/* =====================================================
MAIN CONTAINER
===================================================== */
.block-container {
    padding-top: 1rem !important;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 1rem;
    max-width: 1500px;
}

/* =====================================================
BACKGROUND CIRCUIT EFFECT
===================================================== */
.main::before {
    content: "";
    position: fixed;
    top: 50%;
    left: 50%;
    width: 700px;
    height: 700px;
    transform: translate(-50%, -50%);
    opacity: 0.04;
    background-image:
        linear-gradient(#7c3aed 2px, transparent 2px),
        linear-gradient(90deg, #7c3aed 2px, transparent 2px);
    background-size: 50px 50px;
    z-index: -1;
}

/* =====================================================
SIDEBAR
===================================================== */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255,255,255,0.4);
}

[data-testid="stSidebar"] * {
    color: #111827 !important;
}

/* =====================================================
SIDEBAR LOGO
===================================================== */
.sidebar-logo {
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 35px;
    color: #111827;
}

/* =====================================================
SIDEBAR MENU
===================================================== */
.menu-item {
    padding: 14px 16px;
    border-radius: 14px;
    margin-bottom: 12px;
    font-size: 15px;
    font-weight: 500;
    transition: 0.3s;
    cursor: pointer;
}

.menu-item:hover {
    background: rgba(124,58,237,0.08);
    transform: translateX(4px);
}

/* =====================================================
BUTTONS
===================================================== */
.stButton button {
    width: 100%;
    border-radius: 16px;
    border: none;
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white !important;
    font-weight: 600;
    height: 48px;
    font-size: 15px;
    transition: 0.3s;
    box-shadow: 0px 8px 20px rgba(124,58,237,0.25);
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0px 12px 30px rgba(124,58,237,0.35);
}

/* =====================================================
HERO SECTION
===================================================== */
.hero-container {
    text-align: center;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    color: #111827;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #6B7280;
    font-size: 1rem;
}

/* =====================================================
SEARCH BAR
===================================================== */
.search-box input {
    border-radius: 18px !important;
}

/* =====================================================
MODE BOX
===================================================== */
.mode-box {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.4);
    padding: 18px;
    border-radius: 18px;
    margin-bottom: 30px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.05);
}

/* =====================================================
KPI CARDS
===================================================== */
.kpi-card {
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.4);
    border-radius: 22px;
    padding: 24px;
    min-height: 150px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.06);
    transition: 0.35s;
}

.kpi-card:hover {
    transform: translateY(-6px);
    box-shadow: 0px 18px 35px rgba(124,58,237,0.12);
}

.kpi-title {
    color: #374151;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 12px;
}

.kpi-value {
    color: #111827;
    font-size: 2rem;
    font-weight: 800;
}

.kpi-status {
    color: #22c55e;
    font-size: 13px;
    margin-top: 12px;
}

/* =====================================================
CHAT MESSAGE
===================================================== */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.4);
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.04);
}

/* =====================================================
CHAT INPUT
===================================================== */
.stChatInput > div {
    background: rgba(255,255,255,0.92);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.4);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.06);
    padding: 6px;
}

/* =====================================================
FILE UPLOADER
===================================================== */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.72);
    border-radius: 20px;
    border: 2px dashed #7c3aed;
    padding: 28px;
}

/* =====================================================
RADIO BUTTON
===================================================== */
.stRadio > div {
    background: rgba(255,255,255,0.55);
    border-radius: 14px;
    padding: 14px;
}

/* =====================================================
DATAFRAME
===================================================== */
[data-testid="stDataFrame"] {
    border-radius: 20px;
    overflow: hidden;
}

/* =====================================================
SCROLLBAR
===================================================== */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #7c3aed;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================
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
    st.session_state.chat_mode = "General AI Chat"

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">
    ⚡ Industrial AI
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="menu-item">🏠 Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="menu-item">📂 Files</div>', unsafe_allow_html=True)
    st.markdown('<div class="menu-item">🔔 Notifications</div>', unsafe_allow_html=True)
    st.markdown('<div class="menu-item">📊 Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="menu-item">⚙️ Settings</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("➕ New Analysis"):

        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.session_state.files_processed = False
        st.session_state.report_count = 0
        st.session_state.uploaded_files = []

        st.rerun()

# =====================================================
# HERO SECTION
# =====================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ Industrial AI Assistant</div>
    <div class="hero-subtitle">
        AI-powered Industrial Analytics & RAG Platform
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SEARCH BAR
# =====================================================
st.text_input(
    "",
    placeholder="Search..."
)

# =====================================================
# MODE SELECTION
# =====================================================
st.markdown('<div class="mode-box">', unsafe_allow_html=True)

chat_mode = st.radio(
    "Select AI Mode",
    [
        "General AI Chat",
        "Upload & Analyze Reports"
    ],
    horizontal=True
)

st.session_state.chat_mode = chat_mode

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# KPI SECTION
# =====================================================
report_count = st.session_state.report_count
query_count = len(st.session_state.chat_history)

efficiency = 100 if query_count == 0 else min(100, 95 + query_count)

status = "Connected" if st.session_state.files_processed else "Waiting"

col1, col2, col3, col4 = st.columns(4)

cards = [
    ("📄 REPORTS", report_count),
    ("🧠 AI QUERIES", query_count),
    ("⚡ EFFICIENCY", f"{efficiency}%"),
    ("🛰 STATUS", status)
]

for col, card in zip([col1, col2, col3, col4], cards):

    with col:

        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{card[0]}</div>
            <div class="kpi-value">{card[1]}</div>
            <div class="kpi-status">● Status</div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# FILE UPLOADER
# =====================================================
if st.session_state.chat_mode == "Upload & Analyze Reports":

    st.markdown("## 📂 Upload Industrial Reports")

    uploaded_files = st.file_uploader(
        "Upload reports",
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

                    all_documents.append(
                        Document(
                            page_content=df.to_string(index=False)
                        )
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

            st.success("✅ Files processed successfully!")

        except Exception as e:

            st.error(f"Error: {e}")

# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================
for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])

# =====================================================
# CHAT INPUT
# =====================================================
question = st.chat_input(
    "Ask your Industrial AI question..."
)

# =====================================================
# GENERAL AI CHAT
# =====================================================
if (
    question
    and st.session_state.chat_mode == "General AI Chat"
):

    with st.chat_message("user"):
        st.write(question)

    api_key = st.secrets["GROQ_API_KEY"]

    client = Groq(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": """
You are a highly intelligent industrial AI assistant.

Be modern, helpful, conversational and professional.
"""
        }
    ]

    for chat in st.session_state.chat_history[-5:]:

        messages.append({
            "role": "user",
            "content": chat["question"]
        })

        messages.append({
            "role": "assistant",
            "content": chat["answer"]
        })

    messages.append({
        "role": "user",
        "content": question
    })

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

                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.chat_history.append({
        "question": question,
        "answer": full_response
    })
# ============================================================
# INDUSTRIAL AI WORKSPACE
# PREMIUM AI ANALYTICS PLATFORM
# MODERN CHATGPT STYLE VERSION
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
import time

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Industrial AI Workspace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* =========================================================
APP
========================================================= */

.stApp {

    background: linear-gradient(
        180deg,
        #0F172A 0%,
        #111827 100%
    );

    color: white;
}

/* =========================================================
REMOVE STREAMLIT
========================================================= */

header, footer, #MainMenu {
    visibility: hidden;
}

[data-testid="stHeader"] {
    display: none;
}

/* =========================================================
LAYOUT
========================================================= */

.block-container {

    padding-top: 1.5rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    padding-bottom: 90px;
    max-width: 1450px;
}

/* =========================================================
SIDEBAR
========================================================= */

[data-testid="stSidebar"] {

    background: #0B1120;

    border-right: 1px solid rgba(255,255,255,0.06);

    padding-top: 1rem;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* =========================================================
LOGO
========================================================= */

.sidebar-logo {

    font-size: 28px;
    font-weight: 800;
    color: #8B5CF6;
    margin-bottom: 40px;
    letter-spacing: -1px;
}

/* =========================================================
MENU
========================================================= */

.menu-item {

    padding: 16px 18px;
    border-radius: 18px;
    margin-bottom: 12px;
    font-size: 17px;
    font-weight: 600;
    cursor: pointer;
    transition: 0.3s;
}

.menu-item:hover {

    background: rgba(255,255,255,0.05);

    transform: translateX(4px);
}

/* =========================================================
TITLE
========================================================= */

.title {

    font-size: 48px;
    font-weight: 800;
    color: white;
    margin-top: 10px;
    margin-bottom: 10px;
    letter-spacing: -2px;
    line-height: 1.1;
}

/* =========================================================
WELCOME
========================================================= */

.welcome-text {

    color: #94A3B8;
    font-size: 18px;
    margin-bottom: 30px;
}

/* =========================================================
MODE BOX
========================================================= */

.mode-box {

    background: rgba(255,255,255,0.04);

    backdrop-filter: blur(12px);

    border-radius: 24px;

    padding: 22px;

    border: 1px solid rgba(255,255,255,0.08);

    margin-bottom: 30px;
}

/* =========================================================
RADIO
========================================================= */

.stRadio > div {

    display: flex;

    gap: 16px;
}

.stRadio label {

    background: rgba(255,255,255,0.04) !important;

    border: 1px solid rgba(255,255,255,0.08);

    padding: 16px 22px;

    border-radius: 18px;

    transition: 0.3s;

    font-size: 16px !important;

    font-weight: 600 !important;
}

.stRadio label:hover {

    border: 1px solid #8B5CF6;

    transform: translateY(-3px);
}

/* =========================================================
BUTTONS
========================================================= */

.stButton button {

    width: 100%;

    height: 68px;

    border-radius: 20px;

    border: 1px solid rgba(255,255,255,0.08);

    background: rgba(255,255,255,0.04);

    backdrop-filter: blur(10px);

    color: white !important;

    font-weight: 600;

    font-size: 16px;

    transition: 0.3s;

    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}

.stButton button:hover {

    transform: translateY(-4px);

    border: 1px solid #8B5CF6;

    box-shadow: 0 14px 30px rgba(139,92,246,0.25);
}

/* =========================================================
CHAT MESSAGE
========================================================= */

[data-testid="stChatMessage"] {

    background: rgba(255,255,255,0.04);

    backdrop-filter: blur(12px);

    border-radius: 24px;

    padding: 24px;

    margin-bottom: 18px;

    border: 1px solid rgba(255,255,255,0.08);

    max-width: 920px;

    margin-left: auto;

    margin-right: auto;

    font-size: 17px;

    line-height: 1.7;

    color: white;
}

/* =========================================================
CHAT INPUT
========================================================= */

.stChatInput {

    position: fixed;

    bottom: 20px;

    left: 50%;

    transform: translateX(-50%);

    width: 72%;

    z-index: 999;
}

.stChatInput > div {

    background: rgba(17,24,39,0.95);

    border-radius: 24px;

    border: 1px solid rgba(255,255,255,0.08);

    padding: 14px 18px;

    box-shadow: 0 10px 40px rgba(0,0,0,0.3);

    backdrop-filter: blur(12px);
}

.stChatInput input {

    background: transparent !important;

    color: white !important;

    font-size: 17px !important;

    font-weight: 500;
}

/* =========================================================
FILE UPLOADER
========================================================= */

[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.04);

    border: 2px dashed rgba(139,92,246,0.3);

    border-radius: 24px;

    padding: 35px;

    margin-bottom: 20px;
}

/* =========================================================
DATAFRAME
========================================================= */

[data-testid="stDataFrame"] {

    border-radius: 18px;

    overflow: hidden;

    border: 1px solid rgba(255,255,255,0.08);
}

/* =========================================================
HEADINGS
========================================================= */

h1, h2, h3 {

    color: white !important;
}

/* =========================================================
SCROLLBAR
========================================================= */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.12);
    border-radius: 10px;
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

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "🧠 General AI Chat"

if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = ""

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">
    ⚡ Industrial AI
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="menu-item">💬 New Chat</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="menu-item">📄 Recent Reports</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="menu-item">🧠 AI Workspace</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="menu-item">📚 Documents</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("➕ Start New Chat"):

        st.session_state.chat_history = []

        st.session_state.vectorstore = None

        st.rerun()

# ============================================================
# TITLE
# ============================================================

st.markdown("""
<div class="title">
⚡ Industrial AI Workspace
</div>
""", unsafe_allow_html=True)

# ============================================================
# WELCOME SECTION
# ============================================================

st.markdown("""
<div class="welcome-text">
Analyze industrial reports, detect anomalies, generate AI insights,
and chat intelligently with your uploaded documents.
</div>
""", unsafe_allow_html=True)

# ============================================================
# MODE SELECTION
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
# QUICK ACTIONS
# ============================================================

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    if st.button("⚡ Analyze Reports"):
        st.session_state.quick_prompt = (
            "Analyze uploaded reports and provide key findings."
        )

with col2:
    if st.button("📊 Generate Insights"):
        st.session_state.quick_prompt = (
            "Generate insights from uploaded industrial data."
        )

with col3:
    if st.button("🧠 Summarize Data"):
        st.session_state.quick_prompt = (
            "Summarize uploaded reports clearly."
        )

with col4:
    if st.button("📈 Detect Anomalies"):
        st.session_state.quick_prompt = (
            "Detect anomalies in uploaded reports."
        )

# ============================================================
# FILE UPLOAD
# ============================================================

if st.session_state.chat_mode == "📂 Upload & Analyze Reports":

    st.markdown("## 📂 Upload Industrial Reports")

    uploaded_files = st.file_uploader(
        "Upload Reports",
        type=["pdf", "xlsx", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files:

        try:

            all_documents = []

            for uploaded_file in uploaded_files:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f".{uploaded_file.name.split('.')[-1]}"
                ) as tmp:

                    tmp.write(uploaded_file.read())

                    temp_path = tmp.name

                # ====================================================
                # PDF
                # ====================================================

                if uploaded_file.name.endswith(".pdf"):

                    loader = PyPDFLoader(temp_path)

                    documents = loader.load()

                    all_documents.extend(documents)

                # ====================================================
                # EXCEL
                # ====================================================

                elif uploaded_file.name.endswith(".xlsx"):

                    excel_data = pd.read_excel(
                        temp_path,
                        sheet_name=None
                    )

                    for sheet_name, df in excel_data.items():

                        st.subheader(f"📄 {sheet_name}")

                        st.dataframe(df)

                        numeric_cols = df.select_dtypes(
                            include='number'
                        ).columns

                        if len(numeric_cols) >= 2:

                            fig = px.line(
                                df,
                                x=numeric_cols[0],
                                y=numeric_cols[1],
                                title="Industrial Data Trend"
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                        summary = df.describe().to_string()

                        text = f"""
Sheet Name: {sheet_name}

Data:
{df.to_string(index=False)}

Summary:
{summary}
"""

                        all_documents.append(
                            Document(page_content=text)
                        )

                # ====================================================
                # CSV
                # ====================================================

                elif uploaded_file.name.endswith(".csv"):

                    df = pd.read_csv(temp_path)

                    st.dataframe(df)

                    numeric_cols = df.select_dtypes(
                        include='number'
                    ).columns

                    if len(numeric_cols) >= 2:

                        fig = px.line(
                            df,
                            x=numeric_cols[0],
                            y=numeric_cols[1],
                            title="CSV Trend Analysis"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    summary = df.describe().to_string()

                    text = f"""
CSV Data:
{df.to_string(index=False)}

Summary:
{summary}
"""

                    all_documents.append(
                        Document(page_content=text)
                    )

            # ====================================================
            # CHUNKING
            # ====================================================

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=150
            )

            chunks = splitter.split_documents(all_documents)

            # ====================================================
            # EMBEDDINGS
            # ====================================================

            @st.cache_resource
            def load_embeddings():

                return HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-en-v1.5"
                )

            embeddings = load_embeddings()

            # ====================================================
            # FAISS VECTOR STORE
            # ====================================================

            st.session_state.vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

            st.success("✅ Files processed successfully!")

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
    "Ask anything about industrial reports..."
)

if not question and st.session_state.quick_prompt:
    question = st.session_state.quick_prompt
    st.session_state.quick_prompt = ""

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
You are an advanced industrial AI assistant.

Be intelligent, modern, professional,
and helpful for industrial analytics.
"""
        },
        {
            "role": "user",
            "content": question
        }
    ]

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

                time.sleep(0.01)

                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.chat_history.append({
        "question": question,
        "answer": full_response
    })

# ============================================================
# RAG CHAT
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
You are an advanced industrial AI analyst.

Rules:
- Give professional insights
- Mention trends
- Mention anomalies
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

                time.sleep(0.01)

                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.chat_history.append({
        "question": question,
        "answer": full_response
    })
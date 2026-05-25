# ============================================================
# INDUSTRIAL AI WORKSPACE
# CLEAN PREMIUM CHATGPT STYLE VERSION
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_core.documents import Document
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
# CLEAN CSS
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
    background-color: #0F172A;
    color: white;
}

/* =========================================================
REMOVE STREAMLIT DEFAULTS
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
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* =========================================================
SIDEBAR
========================================================= */

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* =========================================================
LOGO
========================================================= */

.logo-text {
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 40px;
    color: white;
}

/* =========================================================
SIDEBAR MENU
========================================================= */

.menu-item {
    padding: 14px 16px;
    border-radius: 14px;
    margin-bottom: 12px;
    font-size: 16px;
    font-weight: 600;
    background: rgba(255,255,255,0.03);
}

/* =========================================================
TITLE
========================================================= */

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

/* =========================================================
SUBTITLE
========================================================= */

.subtitle {
    text-align: center;
    color: #94A3B8;
    font-size: 18px;
    margin-bottom: 40px;
}

/* =========================================================
BUTTONS
========================================================= */

.stButton button {

    width: 100%;
    height: 95px;

    border-radius: 20px;

    background-color: #1E293B;

    border: 1px solid rgba(255,255,255,0.08);

    color: white;

    font-size: 18px;

    font-weight: 600;

    transition: 0.3s;
}

.stButton button:hover {

    border: 1px solid #8B5CF6;

    transform: translateY(-3px);
}

/* =========================================================
SELECTBOX
========================================================= */

.stSelectbox > div > div {
    background-color: #1E293B;
    color: white;
    border-radius: 14px;
}

/* =========================================================
CHAT MESSAGE
========================================================= */

[data-testid="stChatMessage"] {

    background-color: #111827;

    border-radius: 20px;

    padding: 20px;

    margin-bottom: 16px;

    border: 1px solid rgba(255,255,255,0.06);

    color: white;
}

/* =========================================================
CHAT INPUT
========================================================= */

.stChatInput {
    margin-top: 30px;
}

.stChatInput > div {

    background-color: #111827;

    border-radius: 20px;

    border: 1px solid rgba(255,255,255,0.08);
}

/* =========================================================
FILE UPLOADER
========================================================= */

[data-testid="stFileUploader"] {

    background-color: #111827;

    border: 1px dashed rgba(255,255,255,0.2);

    border-radius: 20px;

    padding: 25px;
}

/* =========================================================
DATAFRAME
========================================================= */

[data-testid="stDataFrame"] {

    border-radius: 18px;

    overflow: hidden;
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

if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = ""

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="logo-text">
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
        '<div class="menu-item">📚 Documents</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="menu-item">⚙️ Settings</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("➕ Start New Chat"):

        st.session_state.chat_history = []

        st.session_state.vectorstore = None

        st.rerun()

# ============================================================
# MAIN TITLE
# ============================================================

st.markdown("""
<div class="main-title">
⚡ Industrial AI Workspace
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Analyze industrial reports, detect anomalies,
generate AI insights, and chat with your documents.
</div>
""", unsafe_allow_html=True)

# ============================================================
# MODE SELECT
# ============================================================

chat_mode = st.selectbox(
    "Select AI Mode",
    [
        "🧠 General AI Chat",
        "📂 Upload & Analyze Reports"
    ]
)

# ============================================================
# QUICK ACTIONS
# ============================================================

st.markdown("### 🚀 Quick Actions")

col1, col2, col3, col4 = st.columns(4)

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

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# FILE UPLOAD
# ============================================================

if chat_mode == "📂 Upload & Analyze Reports":

    st.markdown("## 📂 Upload Industrial Reports")

    uploaded_files = st.file_uploader(
        "Upload PDF / Excel / CSV Files",
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
                # PDF PROCESSING
                # ====================================================

                if uploaded_file.name.endswith(".pdf"):

                    loader = PyPDFLoader(temp_path)

                    documents = loader.load()

                    all_documents.extend(documents)

                # ====================================================
                # EXCEL PROCESSING
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
                                title="Trend Analysis"
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
                # CSV PROCESSING
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
            # FAISS
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

if question and chat_mode == "🧠 General AI Chat":

    with st.chat_message("user"):
        st.write(question)

    api_key = st.secrets["GROQ_API_KEY"]

    client = Groq(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": """
You are an advanced industrial AI assistant.

Be intelligent, professional,
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
    and chat_mode == "📂 Upload & Analyze Reports"
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
- Give accurate insights
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
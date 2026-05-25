# ============================================================
# INDUSTRIAL AI WORKSPACE
# PREMIUM AI ANALYTICS PLATFORM
# COMPLETE FINAL PREMIUM VERSION
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
    background: #F6F8FC;
    color: #111827;
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

    padding-bottom: 140px;

    max-width: 1500px;
}

/* =========================================================
SIDEBAR
========================================================= */

[data-testid="stSidebar"] {

    background: white;

    border-right: 1px solid rgba(0,0,0,0.06);

    padding-top: 1rem;
}

[data-testid="stSidebar"] * {
    color: #111827 !important;
}

/* =========================================================
LOGO
========================================================= */

.sidebar-logo {

    font-size: 30px;

    font-weight: 800;

    color: #4F46E5;

    margin-bottom: 50px;

    letter-spacing: -1px;
}

/* =========================================================
MENU
========================================================= */

.menu-item {

    padding: 18px 18px;

    border-radius: 18px;

    margin-bottom: 14px;

    font-size: 18px;

    font-weight: 600;

    cursor: pointer;

    transition: 0.25s;
}

.menu-item:hover {

    background: #EEF2FF;

    color: #4F46E5;

    transform: translateX(4px);
}

/* =========================================================
BUTTONS PREMIUM
========================================================= */

.stButton button {

    width: 100%;

    height: 70px;

    border-radius: 22px;

    border: none;

    background: white;

    color: #111827 !important;

    font-weight: 700;

    font-size: 18px;

    border: 1px solid rgba(0,0,0,0.06);

    box-shadow: 0 6px 18px rgba(0,0,0,0.04);

    transition: 0.25s;

    padding: 0 24px;
}

.stButton button:hover {

    border: 1px solid #4F46E5;

    color: #4F46E5 !important;

    transform: translateY(-4px);

    box-shadow: 0 14px 30px rgba(79,70,229,0.12);
}

/* =========================================================
TITLE
========================================================= */

.title {

    font-size: 72px;

    font-weight: 800;

    color: #111827;

    margin-top: 10px;

    margin-bottom: 40px;

    letter-spacing: -3px;

    line-height: 1.05;
}

/* =========================================================
MODE BOX
========================================================= */

.mode-box {

    background: white;

    border-radius: 24px;

    padding: 22px 24px;

    border: 1px solid rgba(0,0,0,0.06);

    margin-bottom: 35px;

    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
}

/* =========================================================
RADIO PREMIUM
========================================================= */

.stRadio > div {

    display: flex;

    gap: 18px;
}

.stRadio label {

    background: white !important;

    border: 1px solid rgba(0,0,0,0.08);

    padding: 18px 26px;

    border-radius: 18px;

    transition: 0.25s;

    box-shadow: 0 4px 14px rgba(0,0,0,0.03);

    font-size: 18px !important;

    font-weight: 700 !important;
}

.stRadio label:hover {

    border: 1px solid #4F46E5;

    transform: translateY(-3px);

    box-shadow: 0 12px 24px rgba(79,70,229,0.10);
}

/* =========================================================
CHAT
========================================================= */

[data-testid="stChatMessage"] {

    background: white;

    border-radius: 28px;

    padding: 26px;

    margin-bottom: 22px;

    border: 1px solid rgba(0,0,0,0.05);

    box-shadow: 0 4px 18px rgba(0,0,0,0.04);

    max-width: 980px;

    margin-left: auto;

    margin-right: auto;

    font-size: 18px;

    line-height: 1.8;
}

/* =========================================================
CHAT INPUT
========================================================= */

.stChatInput {

    position: fixed;

    bottom: 24px;

    left: 50%;

    transform: translateX(-50%);

    width: 74%;

    z-index: 999;
}

.stChatInput > div {

    background: white;

    border-radius: 28px;

    border: 1px solid rgba(0,0,0,0.08);

    padding: 14px 18px;

    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

.stChatInput input {

    background: transparent !important;

    color: #111827 !important;

    font-size: 18px !important;

    font-weight: 500;
}

/* =========================================================
UPLOAD BOX
========================================================= */

[data-testid="stFileUploader"] {

    background: white;

    border: 2px dashed rgba(79,70,229,0.2);

    border-radius: 28px;

    padding: 40px;

    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
}

[data-testid="stFileUploader"] small {

    font-size: 16px !important;

    color: #6B7280 !important;
}

[data-testid="stFileUploader"] section {

    font-size: 18px !important;
}

/* =========================================================
DATAFRAME
========================================================= */

[data-testid="stDataFrame"] {

    border-radius: 20px;

    overflow: hidden;

    border: 1px solid rgba(0,0,0,0.06);
}

/* =========================================================
SUBHEADERS
========================================================= */

h1, h2, h3 {

    font-size: 34px !important;

    font-weight: 700 !important;

    color: #111827 !important;
}

/* =========================================================
REMOVE LINES
========================================================= */

hr {
    display: none;
}

/* =========================================================
SCROLLBAR
========================================================= */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: rgba(0,0,0,0.12);
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

if "files_processed" not in st.session_state:
    st.session_state.files_processed = False

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

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

        st.session_state.files_processed = False

        st.session_state.uploaded_files = []

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

        st.session_state.uploaded_files = uploaded_files

        try:

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

                    for sheet_name, df in excel_data.items():

                        st.subheader(f"📄 Sheet: {sheet_name}")

                        st.dataframe(df)

                        numeric_cols = df.select_dtypes(
                            include='number'
                        ).columns

                        if len(numeric_cols) >= 2:

                            st.subheader("📈 Trend Analysis")

                            fig1 = px.line(
                                df,
                                x=numeric_cols[0],
                                y=numeric_cols[1],
                                title="Industrial Trend Analysis"
                            )

                            st.plotly_chart(
                                fig1,
                                use_container_width=True
                            )

                            st.subheader("📊 Consumption Analysis")

                            fig2 = px.bar(
                                df,
                                x=numeric_cols[0],
                                y=numeric_cols[1],
                                title="Consumption Comparison"
                            )

                            st.plotly_chart(
                                fig2,
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
                            title="CSV Data Analysis"
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

Be intelligent, modern and professional.
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

                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.chat_history.append({
        "question": question,
        "answer": full_response
    })

# ============================================================
# RAG RESPONSE
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
- Be conversational
- Give accurate insights
- Mention anomalies
- Mention trends
- Give recommendations
- Answer ONLY from context
- Do NOT generate fake image links

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

                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.chat_history.append({
        "question": question,
        "answer": full_response
    })
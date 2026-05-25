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

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Industrial AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================
# CUSTOM CSS
# =========================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* =========================================
MAIN APP
========================================= */
.stApp {
    background: #F4F7FB;
}

/* =========================================
REMOVE STREAMLIT DEFAULT HEADER
========================================= */
header {
    visibility: hidden;
    height: 0px;
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

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

/* =========================================
REMOVE TOP GAP
========================================= */
.block-container {
    padding-top: 0rem !important;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* =========================================
SIDEBAR
========================================= */
[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid #E5E7EB;
}

[data-testid="stSidebar"] * {
    color: #111827 !important;
}

/* =========================================
SIDEBAR TITLE
========================================= */
.sidebar-logo {
    font-size: 26px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 30px;
}

/* =========================================
MENU ITEMS
========================================= */
.menu-item {
    padding: 12px 14px;
    border-radius: 12px;
    margin-bottom: 10px;
    font-weight: 500;
    color: #374151;
    transition: 0.3s;
    cursor: pointer;
}

.menu-item:hover {
    background: #F3F4F6;
}

/* =========================================
BUTTONS
========================================= */
.stButton button {
    width: 100%;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #14B8A6, #06B6D4);
    color: white !important;
    font-weight: 600;
    height: 45px;
    transition: 0.3s;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.12);
}

/* =========================================
HERO SECTION
========================================= */
.hero-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    color: #374151;
    margin-top: 10px;
}

.hero-subtitle {
    text-align: center;
    color: #6B7280;
    margin-bottom: 35px;
    font-size: 1.1rem;
}

/* =========================================
MODE BOX
========================================= */
.mode-box {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #E5E7EB;
    margin-bottom: 25px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.04);
}

/* =========================================
KPI CARDS
========================================= */
.kpi-card {
    background: white;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.04);
    transition: 0.3s;
}

.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0px 10px 28px rgba(0,0,0,0.08);
}

.kpi-title {
    color: #6B7280;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 10px;
}

.kpi-value {
    color: #14B8A6;
    font-size: 32px;
    font-weight: 700;
}

/* =========================================
CHAT MESSAGE
========================================= */
[data-testid="stChatMessage"] {
    background: white;
    border-radius: 16px;
    border: 1px solid #E5E7EB;
    padding: 16px;
    margin-bottom: 15px;
}

/* =========================================
CHAT INPUT
========================================= */
.stChatInput > div {
    background: white;
    border-radius: 16px;
    border: 1px solid #D1D5DB;
}

/* =========================================
FILE UPLOADER
========================================= */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 18px;
    border: 2px dashed #14B8A6;
    padding: 25px;
}

/* =========================================
RADIO BUTTONS
========================================= */
.stRadio > div {
    background: #F9FAFB;
    padding: 15px;
    border-radius: 14px;
}

/* =========================================
DATAFRAME
========================================= */
[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid #E5E7EB;
}

/* =========================================
SCROLLBAR
========================================= */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #14B8A6;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# SESSION STATE
# =========================================
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

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">
    ⚡ INDUSTRIAL AI
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

# =========================================
# HERO SECTION
# =========================================
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

# =========================================
# MODE SELECTION
# =========================================
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

# =========================================
# KPI SECTION
# =========================================
report_count = st.session_state.report_count
query_count = len(st.session_state.chat_history)

efficiency = 100 if query_count == 0 else min(100, 95 + query_count)

status = "Connected" if st.session_state.files_processed else "Waiting"

col1, col2, col3, col4 = st.columns(4)

cards = [
    ("📄 Reports", report_count),
    ("🤖 AI Queries", query_count),
    ("⚡ Efficiency", f"{efficiency}%"),
    ("🟢 Status", status)
]

for col, card in zip([col1, col2, col3, col4], cards):

    with col:

        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{card[0]}</div>
            <div class="kpi-value">{card[1]}</div>
        </div>
        """, unsafe_allow_html=True)

# =========================================
# FILE UPLOADER
# =========================================
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

# =========================================
# ANALYTICS
# =========================================
if (
    st.session_state.files_processed
    and st.session_state.chat_mode == "Upload & Analyze Reports"
):

    st.markdown("## 📊 Analytics Dashboard")

    try:

        uploaded_files = st.session_state.uploaded_files

        for uploaded_file in uploaded_files:

            uploaded_file.seek(0)

            if uploaded_file.name.endswith(".csv"):

                df = pd.read_csv(uploaded_file)

            elif uploaded_file.name.endswith(".xlsx"):

                df = pd.read_excel(uploaded_file)

            else:
                continue

            st.markdown(f"### 📄 {uploaded_file.name}")

            st.dataframe(df.head())

            numeric_cols = df.select_dtypes(include='number').columns

            if len(numeric_cols) > 0:

                target_col = numeric_cols[0]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Average",
                        round(df[target_col].mean(), 2)
                    )

                with col2:
                    st.metric(
                        "Maximum",
                        round(df[target_col].max(), 2)
                    )

                with col3:
                    st.metric(
                        "Minimum",
                        round(df[target_col].min(), 2)
                    )

                fig = px.line(
                    df,
                    y=target_col,
                    title=f"{target_col} Trend Analysis"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    except Exception as e:

        st.error(f"Analytics unavailable: {e}")

# =========================================
# DISPLAY CHAT HISTORY
# =========================================
for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])

# =========================================
# CHAT INPUT
# =========================================
question = st.chat_input(
    "Ask your industrial AI question..."
)

# =========================================
# GENERAL AI CHAT
# =========================================
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
You are a modern AI assistant.

Be conversational, intelligent and helpful.
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

# =========================================
# RAG AI CHAT
# =========================================
if (
    question
    and st.session_state.vectorstore is not None
    and st.session_state.chat_mode == "Upload & Analyze Reports"
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
- Answer ONLY from context
- Be conversational
- Give insights
- Explain clearly
- Suggest improvements

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
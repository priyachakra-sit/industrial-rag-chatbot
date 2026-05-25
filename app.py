# ============================================================
# INDUSTRIAL AI WORKSPACE
# PREMIUM LIGHT MODERN AI UI
# ============================================================

import pandas as pd
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
# PREMIUM LIGHT UI CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

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
MAIN LAYOUT
========================================================= */

.block-container {

    padding-top: 1.5rem;

    padding-left: 2rem;

    padding-right: 2rem;

    padding-bottom: 120px;

    max-width: 1450px;
}

/* =========================================================
SIDEBAR
========================================================= */

[data-testid="stSidebar"] {

    background: white;

    border-right: 1px solid rgba(0,0,0,0.06);
}

[data-testid="stSidebar"] * {
    color: #111827 !important;
}

/* =========================================================
LOGO
========================================================= */

.sidebar-logo {

    font-size: 24px;

    font-weight: 700;

    color: #4F46E5;

    margin-bottom: 40px;
}

/* =========================================================
MENU ITEMS
========================================================= */

.menu-item {

    padding: 14px 16px;

    border-radius: 14px;

    margin-bottom: 10px;

    font-size: 15px;

    font-weight: 500;

    cursor: pointer;

    transition: 0.2s;
}

.menu-item:hover {

    background: #EEF2FF;

    color: #4F46E5;

    transform: translateX(3px);
}

/* =========================================================
BUTTONS
========================================================= */

.stButton button {

    width: 100%;

    height: 46px;

    border-radius: 14px;

    border: none;

    background: #4F46E5;

    color: white !important;

    font-weight: 600;

    box-shadow: 0 4px 14px rgba(79,70,229,0.18);
}

.stButton button:hover {

    background: #4338CA;
}

/* =========================================================
WELCOME
========================================================= */

.welcome-title {

    font-size: 42px;

    font-weight: 700;

    color: #111827;

    margin-top: 20px;

    margin-bottom: 10px;
}

.welcome-subtitle {

    color: #6B7280;

    font-size: 18px;

    margin-bottom: 35px;
}

/* =========================================================
MODE BOX
========================================================= */

.mode-box {

    background: white;

    border-radius: 18px;

    padding: 12px 18px;

    border: 1px solid rgba(0,0,0,0.06);

    margin-bottom: 25px;

    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
}

/* =========================================================
RADIO BUTTONS
========================================================= */

.stRadio > div {
    background: transparent !important;
    border: none !important;
}

/* =========================================================
QUICK ACTIONS
========================================================= */

.quick-action {

    background: white;

    border-radius: 18px;

    padding: 18px;

    border: 1px solid rgba(0,0,0,0.05);

    text-align: center;

    font-weight: 600;

    margin-bottom: 20px;

    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
}

/* =========================================================
CHAT MESSAGES
========================================================= */

[data-testid="stChatMessage"] {

    background: white;

    border-radius: 22px;

    padding: 18px;

    margin-bottom: 18px;

    border: 1px solid rgba(0,0,0,0.05);

    box-shadow: 0 2px 10px rgba(0,0,0,0.03);

    max-width: 900px;

    margin-left: auto;

    margin-right: auto;
}

/* =========================================================
CHAT INPUT
========================================================= */

.stChatInput {

    position: fixed;

    bottom: 20px;

    left: 50%;

    transform: translateX(-50%);

    width: 70%;

    z-index: 999;
}

.stChatInput > div {

    background: white;

    border-radius: 24px;

    border: 1px solid rgba(0,0,0,0.08);

    padding: 10px 14px;

    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
}

.stChatInput input {

    background: transparent !important;

    color: #111827 !important;

    font-size: 15px;
}

/* =========================================================
FILE UPLOADER
========================================================= */

[data-testid="stFileUploader"] {

    background: white;

    border: 2px dashed rgba(79,70,229,0.2);

    border-radius: 20px;

    padding: 30px;

    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
}

/* =========================================================
SUCCESS MESSAGE
========================================================= */

.stSuccess {

    background: #ECFDF5 !important;

    color: #065F46 !important;

    border-radius: 12px;
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
# WELCOME SECTION
# ============================================================

st.markdown("""
<div class="welcome-title">
Industrial AI Workspace
</div>

<div class="welcome-subtitle">
Analyze reports, detect anomalies, generate insights and interact with your AI assistant.
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
    st.markdown(
        '<div class="quick-action">⚡ Analyze Reports</div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        '<div class="quick-action">📊 Generate Insights</div>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        '<div class="quick-action">🧠 Summarize Data</div>',
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        '<div class="quick-action">📈 Detect Anomalies</div>',
        unsafe_allow_html=True
    )

# ============================================================
# FILE UPLOAD
# ============================================================

if st.session_state.chat_mode == "📂 Upload & Analyze Reports":

    st.markdown("### 📂 Upload Industrial Reports")

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
    "Ask anything about industrial reports, analytics or insights..."
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
You are an advanced industrial AI assistant.

Be intelligent, conversational, professional and modern.
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
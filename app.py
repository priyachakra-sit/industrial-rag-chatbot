# ============================================================
# INDUSTRIAL AI WORKSPACE
# PREMIUM LIGHT EMERALD GLASS UI
# FULLY WORKING FINAL VERSION
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
    page_title="Industrial AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PREMIUM LIGHT EMERALD GLASS CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* =========================================================
BACKGROUND
========================================================= */

.stApp {

    background:
    linear-gradient(
        rgba(240,255,248,0.82),
        rgba(228,248,238,0.88)
    ),
    url("https://images.unsplash.com/photo-1498050108023-c5249f4df085?q=80&w=1974&auto=format&fit=crop");

    background-size: cover;

    background-position: center;

    background-attachment: fixed;

    color: #103B2C;
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

    padding-top: 1rem;

    padding-left: 2rem;

    padding-right: 2rem;

    padding-bottom: 170px;

    max-width: 1500px;
}

/* =========================================================
MAIN GLASS PANEL
========================================================= */

.main > div {

    background: rgba(255,255,255,0.28);

    backdrop-filter: blur(20px);

    border: 1px solid rgba(255,255,255,0.55);

    border-radius: 30px;

    padding: 25px;

    box-shadow:
    0 10px 45px rgba(16,59,44,0.10),
    0 0 30px rgba(0,180,110,0.06),
    inset 0 1px 1px rgba(255,255,255,0.7),
    inset 0 -1px 1px rgba(255,255,255,0.15);
}

/* =========================================================
SIDEBAR
========================================================= */

[data-testid="stSidebar"] {

    background: rgba(255,255,255,0.30);

    backdrop-filter: blur(18px);

    border-right: 1px solid rgba(255,255,255,0.45);
}

[data-testid="stSidebar"] * {
    color: #103B2C !important;
}

/* =========================================================
SIDEBAR LOGO
========================================================= */

.sidebar-logo {

    font-size: 32px;

    font-weight: 800;

    color: #0F5132;

    margin-top: 20px;

    margin-bottom: 80px;

    letter-spacing: -1px;

    text-shadow: 0 0 12px rgba(0,180,110,0.18);
}

/* =========================================================
PREMIUM GLOW BUTTONS
========================================================= */

.stButton button {

    position: relative;

    overflow: hidden;

    width: 100%;

    height: 65px;

    border-radius: 22px;

    border: 1px solid rgba(255,255,255,0.55);

    background: rgba(255,255,255,0.24);

    backdrop-filter: blur(12px);

    color: #103B2C !important;

    font-weight: 700;

    font-size: 17px;

    box-shadow:
    0 6px 18px rgba(16,59,44,0.08),
    0 0 15px rgba(0,180,110,0.08),
    inset 0 1px 1px rgba(255,255,255,0.7);

    transition:
    transform 0.25s ease,
    box-shadow 0.25s ease,
    border 0.25s ease;
}

.stButton button::before {

    content: "";

    position: absolute;

    top: 0;

    left: -120%;

    width: 80%;

    height: 100%;

    background:
    linear-gradient(
        120deg,
        transparent,
        rgba(255,255,255,0.55),
        transparent
    );

    transform: skewX(-25deg);

    transition: 0.7s;
}

.stButton button:hover {

    transform: translateY(-3px) scale(1.01);

    border: 1px solid rgba(0,180,110,0.45);

    box-shadow:
    0 12px 30px rgba(0,180,110,0.16),
    0 0 24px rgba(0,180,110,0.18),
    inset 0 1px 1px rgba(255,255,255,0.8);
}

.stButton button:hover::before {

    left: 140%;
}

/* =========================================================
RADIO BUTTONS
========================================================= */

.stRadio > div {

    display: flex;

    gap: 22px;

    margin-bottom: 30px;
}

.stRadio label {

    background: rgba(255,255,255,0.34) !important;

    border: 1px solid rgba(255,255,255,0.55);

    padding: 16px 24px;

    border-radius: 20px;

    backdrop-filter: blur(12px);

    color: #103B2C !important;

    box-shadow:
    0 6px 18px rgba(16,59,44,0.06),
    inset 0 1px 1px rgba(255,255,255,0.7);

    transition: 0.25s;
}

/* =========================================================
CHAT ANIMATION
========================================================= */

@keyframes smoothFadeIn {

    from {

        opacity: 0;

        transform: translateY(18px) scale(0.98);
    }

    to {

        opacity: 1;

        transform: translateY(0px) scale(1);
    }
}

/* =========================================================
CHAT MESSAGE
========================================================= */

[data-testid="stChatMessage"] {

    animation: smoothFadeIn 0.45s ease;

    background: rgba(255,255,255,0.34);

    border: 1px solid rgba(255,255,255,0.55);

    backdrop-filter: blur(16px);

    border-radius: 26px;

    padding: 22px;

    margin-bottom: 18px;

    color: #103B2C;

    box-shadow:
    0 8px 24px rgba(16,59,44,0.06),
    inset 0 1px 1px rgba(255,255,255,0.75);

    max-width: 980px;

    margin-left: auto;

    margin-right: auto;
}

/* =========================================================
CHAT INPUT
========================================================= */

.stChatInput {

    position: fixed;

    bottom: 24px;

    left: calc(50% + 60px);

    transform: translateX(-50%);

    width: calc(100% - 420px);

    max-width: 1050px;

    z-index: 999;
}

.stChatInput > div {

    background: rgba(255,255,255,0.50);

    backdrop-filter: blur(18px);

    border: 1px solid rgba(255,255,255,0.75);

    border-radius: 34px;

    padding: 14px 18px;

    box-shadow:
    0 12px 30px rgba(16,59,44,0.10),
    0 0 20px rgba(0,180,110,0.08),
    inset 0 1px 1px rgba(255,255,255,0.9),
    inset 0 -1px 1px rgba(255,255,255,0.25);
}

.stChatInput input {

    background: transparent !important;

    color: #103B2C !important;

    font-size: 17px !important;

    font-weight: 500;
}

/* =========================================================
UPLOAD BOX
========================================================= */

[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.30);

    border: 1px dashed rgba(0,180,110,0.28);

    border-radius: 26px;

    backdrop-filter: blur(12px);

    padding: 40px;

    color: #103B2C !important;
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

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    if st.button("➕ Start New Chat"):

        st.session_state.chat_history = []

        st.session_state.vectorstore = None

        st.session_state.files_processed = False

        st.session_state.uploaded_files = []

        st.rerun()

# ============================================================
# MODE SELECTION
# ============================================================

chat_mode = st.radio(
    "",
    [
        "🧠 General AI Chat",
        "📂 Upload & Analyze Reports"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.session_state.chat_mode = chat_mode

# ============================================================
# QUICK ACTIONS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("⚡ Analyze Reports"):
        st.session_state.quick_prompt = "Analyze uploaded reports"

with col2:
    if st.button("📊 Generate Insights"):
        st.session_state.quick_prompt = "Generate insights"

with col3:
    if st.button("🧠 Summarize Data"):
        st.session_state.quick_prompt = "Summarize uploaded reports"

with col4:
    if st.button("📈 Detect Anomalies"):
        st.session_state.quick_prompt = "Detect anomalies"

# ============================================================
# FILE UPLOAD + RAG
# ============================================================

if st.session_state.chat_mode == "📂 Upload & Analyze Reports":

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

                        st.dataframe(df)

                        text = df.to_string(index=False)

                        all_documents.append(
                            Document(page_content=text)
                        )

                elif uploaded_file.name.endswith(".csv"):

                    df = pd.read_csv(temp_path)

                    st.dataframe(df)

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

            st.success("✅ Files processed successfully!")

        except Exception as e:

            st.error(f"Error: {e}")

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
    "Ask anything..."
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

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ],
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
# RAG CHATBOT
# ============================================================

if (
    question
    and st.session_state.vectorstore is not None
    and st.session_state.chat_mode
    == "📂 Upload & Analyze Reports"
):

    docs = st.session_state.vectorstore.similarity_search(
        question,
        k=5
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Context:
{context}

Question:
{question}
"""

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

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
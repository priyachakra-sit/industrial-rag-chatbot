# ============================================================
# INDUSTRIAL AI WORKSPACE
# FINAL STABLE VERSION
# MEMORY + WEB SEARCH + RAG + PREMIUM UI
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

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="InsightForge AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PREMIUM GLASS UI
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

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

header, footer, #MainMenu {
    visibility: hidden;
}

[data-testid="stHeader"] {
    display: none;
}

.block-container {
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 170px;
    max-width: 1500px;
}

.main > div {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.55);
    border-radius: 30px;
    padding: 25px;
}

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.30);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(255,255,255,0.45);
}

[data-testid="stSidebar"] * {
    color: #103B2C !important;
}

.stButton button {
    width: 100%;
    height: 60px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.55);
    background: rgba(255,255,255,0.24);
    backdrop-filter: blur(12px);
    color: #103B2C !important;
    font-weight: 700;
}

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.34);
    border: 1px solid rgba(255,255,255,0.55);
    backdrop-filter: blur(16px);
    border-radius: 26px;
    padding: 22px;
    margin-bottom: 18px;
    color: #103B2C;
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
}

.stChatInput {
    position: fixed;
    bottom: 30px;
    width: 75%;
    left: 50%;
    transform: translateX(-50%);
}

.stChatInput > div {
    border-radius: 40px;
    background: white;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# API CLIENTS
# ============================================================

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

tavily = TavilyClient(
    api_key=st.secrets["TAVILY_API_KEY"]
)

# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = ""

if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a modern conversational AI assistant like ChatGPT.

RULES:
- Speak naturally and casually
- Keep responses concise and clean
- Maintain conversation memory
- Avoid sounding like an interviewer
- Do NOT ask follow-up questions unless necessary
- Never ask follow-up questions after greetings or introductions
- Avoid ending every response with a question
- Prefer concise explanations unless detailed depth is needed
- For simple introductions, reply briefly
- Avoid unnecessary conversation extension
- Use structure only for technical topics
- Be intelligent, modern, and human-like
- Use latest web information when needed
"""

# ============================================================
# DATA ANALYSIS
# ============================================================

def advanced_dataframe_analysis(df):

    analysis = {}

    analysis["rows"] = df.shape[0]
    analysis["columns"] = df.shape[1]

    missing = df.isnull().sum()

    analysis["missing_values"] = (
        missing[missing > 0].to_dict()
    )

    numeric_df = df.select_dtypes(include='number')

    if not numeric_df.empty:

        analysis["summary_statistics"] = (
            numeric_df.describe().to_dict()
        )

    return analysis

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div style='text-align:center'>

<h1 style='font-size:60px;
color:#0F5132;
font-weight:800;'>

⚡ InsightForge AI

</h1>

<h3 style='color:#5B6B65;'>

Your Intelligent Industrial Assistant

</h3>

</div>
""",
unsafe_allow_html=True)

# ============================================================
# FILE UPLOAD
# ============================================================

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_files = st.file_uploader(
        "",
        type=["pdf", "xlsx", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

# ============================================================
# SHOW WELCOME MESSAGE ONLY WHEN NO FILES AND NO CHAT
# ============================================================

if not uploaded_files and len(st.session_state.chat_history) == 0:

    st.markdown(
        """
        <div style="
        text-align:center;
        margin-top:80px;
        color:#0F5132;
        ">

        <div style='font-size:90px;'>💬</div>

        <h2>Start a Conversation</h2>

        <p>
        Upload your reports and ask questions
        to get insights and answers.<br>
        Or just ask me anything!
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# PROCESS UPLOADED FILES
# ============================================================

if uploaded_files:

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

                analysis = advanced_dataframe_analysis(df)

                text = f"""
DATASET:
{df.head(100).to_string(index=False)}

ANALYSIS:
{analysis}
"""

                all_documents.append(
                    Document(page_content=text)
                )

        elif uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(temp_path)
            analysis = advanced_dataframe_analysis(df)

            text = f"""
DATASET:
{df.head(100).to_string(index=False)}

ANALYSIS:
{analysis}
"""

            all_documents.append(
                Document(page_content=text)
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250
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

    st.toast("Reports uploaded successfully ✅")

# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

recent_history = st.session_state.chat_history[-4:]

for chat in recent_history:

    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])

# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about uploaded reports or just chat..."
)

# ============================================================
# HANDLE QUESTION — RAG or NORMAL CHAT
# ============================================================

if question:

    with st.chat_message("user"):
        st.write(question)

    # FILE UPLOADED → RAG
    if st.session_state.vectorstore is not None:

        docs = st.session_state.vectorstore.similarity_search(
            question,
            k=4
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
DATA CONTEXT:
{context}

QUESTION:
{question}

RULES:
1. Answer only the user's question.
2. Use only the uploaded data.
3. Never hallucinate.
4. Do not generate executive summaries unless asked.
5. Do not generate recommendations unless asked.
6. Keep responses concise.
7. If data is unavailable, say "Data not available".
"""

    # NO FILE → NORMAL CHAT
    else:

        prompt = question

    # Build message history for context
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    for chat in st.session_state.chat_history[-4:]:
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
        "content": prompt
    })

    with st.chat_message("assistant"):

        placeholder = st.empty()
        full_response = ""

        try:

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1,
                max_tokens=1000,
                stream=True
            )

            for chunk in response:

                delta = chunk.choices[0].delta.content

                if delta:
                    full_response += delta
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        except Exception:

            placeholder.error(
                "⚠️ AI service is temporarily busy. Please try again in a few seconds."
            )

    if full_response:

        st.session_state.chat_history.append({
            "question": question,
            "answer": full_response
        })
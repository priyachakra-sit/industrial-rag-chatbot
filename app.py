
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
    page_title="Industrial AI",
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
    background: rgba(255,255,255,0.28);
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
    max-width: 980px;
    margin-left: auto;
    margin-right: auto;
}

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
# WEB SEARCH
# ============================================================

def web_search(query):

    try:

        results = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=2
        )

        web_context = ""

        for result in results["results"]:

            web_context += f"""

TITLE:
{result['title']}

CONTENT:
{result['content']}

URL:
{result['url']}

"""

        return web_context

    except:

        return ""

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
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚡ Industrial AI")

    if st.button("➕ Start New Chat"):

        st.session_state.chat_history = []
        st.session_state.vectorstore = None

        st.rerun()

# ============================================================
# MODE
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

# ============================================================
# QUICK BUTTONS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    if st.button("⚡ Analyze Reports"):

        st.session_state.quick_prompt = (
            "Analyze the uploaded reports deeply."
        )

        st.session_state.analysis_ready = True

with col2:

    if st.button("📊 Generate Insights"):

        st.session_state.quick_prompt = (
            "Generate insights from uploaded data."
        )

        st.session_state.analysis_ready = True

with col3:

    if st.button("🧠 Summarize Data"):

        st.session_state.quick_prompt = (
            "Summarize the uploaded reports."
        )

        st.session_state.analysis_ready = True

with col4:

    if st.button("📈 Detect Anomalies"):

        st.session_state.quick_prompt = (
            "Detect anomalies in uploaded data."
        )

        st.session_state.analysis_ready = True

# ============================================================
# FILE UPLOAD
# ============================================================

if chat_mode == "📂 Upload & Analyze Reports":

    uploaded_files = st.file_uploader(
        "Upload Reports",
        type=["pdf", "xlsx", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files and st.session_state.analysis_ready:

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

                    st.subheader(f"📄 {sheet_name}")
                    st.dataframe(df)

                    analysis = advanced_dataframe_analysis(df)

                    text = f"""

DATASET:
{df.to_string(index=False)}

ANALYSIS:
{analysis}

"""

                    all_documents.append(
                        Document(page_content=text)
                    )

            elif uploaded_file.name.endswith(".csv"):

                df = pd.read_csv(temp_path)

                st.dataframe(df)

                analysis = advanced_dataframe_analysis(df)

                text = f"""

DATASET:
{df.to_string(index=False)}

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

        st.success("✅ Analysis Engine Ready!")

        st.session_state.analysis_ready = False

# ============================================================
# DISPLAY CHAT
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

question = None

if (
    "quick_prompt" in st.session_state
    and st.session_state.quick_prompt
):

    question = st.session_state.quick_prompt

    st.session_state.quick_prompt = ""

user_input = st.chat_input(
    "Ask anything..."
)

if user_input:

    question = user_input

# ============================================================
# GENERAL AI CHAT
# ============================================================

if (
    question
    and chat_mode == "🧠 General AI Chat"
):

    with st.chat_message("user"):

        st.write(question)

    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        latest_keywords = [
    "latest",
    "news",
    "today",
    "current",
    "recent",
    "update",
    "2025",
    "trend"
]

        if any(word in question.lower() for word in latest_keywords):

            web_context = web_search(question)

        else:

            web_context = ""

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        recent_history = st.session_state.chat_history[-4:]

        for chat in recent_history:

            messages.append({
                "role": "user",
                "content": chat["question"]
            })

            messages.append({
                "role": "assistant",
                "content": chat["answer"]
            })

        if web_context:

            messages.append({
                "role": "system",
                "content": f"""

LATEST WEB INFORMATION:

{web_context}

"""
            })

        messages.append({
            "role": "user",
            "content": question
        })

        try:

            response = client.chat.completions.create(

                model="llama-3.1-8b-instant",

                messages=messages,

                temperature=0.1,

                max_tokens=1200,

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

        except Exception:

            placeholder.error(
                "⚠️ AI service is temporarily busy. Please try again in a few seconds."
            )

    if full_response:

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
    and chat_mode == "📂 Upload & Analyze Reports"
):

    with st.chat_message("user"):

        st.write(question)

    docs = st.session_state.vectorstore.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""

You are an advanced industrial AI analytics engine.

DATA CONTEXT:
{context}

QUESTION:
{question}

RULES:
1. Mention trends
2. Mention anomalies
3. Mention recommendations
4. Never hallucinate
5. Use only provided data
6. Keep answers concise
7. Focus on actionable insights

"""

    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        try:

            response = client.chat.completions.create(

                model="llama-3.1-8b-instant",

                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.1,

                max_tokens=1000,

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

        except Exception:

            placeholder.error(
                "⚠️ AI service is temporarily busy. Please try again in a few seconds."
            )

    if full_response:

        st.session_state.chat_history.append({

            "question": question,

            "answer": full_response
        })


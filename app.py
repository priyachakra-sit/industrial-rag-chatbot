# ============================================================
# INDUSTRIAL AI WORKSPACE
# ULTRA AI VERSION
# MEMORY + WEB SEARCH + RAG + SEQUENCE AI
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
from tavily import TavilyClient
import tempfile
import numpy as np

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
You are an advanced AI assistant.

RULES:
1. Always answer step-by-step
2. Maintain conversation continuity
3. Never break sequence
4. Remember previous messages
5. Use headings and bullet points
6. Give structured responses
7. Answer intelligently
8. Use latest web information if available
9. If user asks multiple things:
   answer in exact sequence
10. Be highly professional
"""

# ============================================================
# WEB SEARCH
# ============================================================

def web_search(query):

    try:

        results = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5
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
# ADVANCED DATA ANALYSIS ENGINE
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

        trends = {}

        for col in numeric_df.columns:

            try:

                first = numeric_df[col].iloc[0]
                last = numeric_df[col].iloc[-1]

                if last > first:
                    trends[col] = "Increasing"

                elif last < first:
                    trends[col] = "Decreasing"

                else:
                    trends[col] = "Stable"

            except:

                trends[col] = "Unknown"

        analysis["trends"] = trends

        anomalies = {}

        for col in numeric_df.columns:

            mean = numeric_df[col].mean()

            std = numeric_df[col].std()

            upper = mean + (2 * std)

            lower = mean - (2 * std)

            outliers = numeric_df[
                (numeric_df[col] > upper)
                | (numeric_df[col] < lower)
            ]

            anomalies[col] = len(outliers)

        analysis["anomalies"] = anomalies

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
# QUICK ACTION BUTTONS
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

            # PDF

            if uploaded_file.name.endswith(".pdf"):

                loader = PyPDFLoader(temp_path)

                documents = loader.load()

                all_documents.extend(documents)

            # XLSX

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

            # CSV

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

for chat in st.session_state.chat_history:

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

        # ====================================================
        # WEB SEARCH
        # ====================================================

        web_context = web_search(question)

        # ====================================================
        # MEMORY SYSTEM
        # ====================================================

        messages = [

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }

        ]

        # ====================================================
        # CHAT HISTORY MEMORY
        # ====================================================

        for chat in st.session_state.chat_history:

            messages.append({
                "role": "user",
                "content": chat["question"]
            })

            messages.append({
                "role": "assistant",
                "content": chat["answer"]
            })

        # ====================================================
        # WEB CONTEXT
        # ====================================================

        if web_context:

            messages.append({

                "role": "system",

                "content": f"""

LATEST WEB INFORMATION:

{web_context}

"""
            })

        # ====================================================
        # CURRENT QUESTION
        # ====================================================

        messages.append({

            "role": "user",

            "content": question
        })

        # ====================================================
        # DEBUG CHECK
        # ====================================================

        

        # ====================================================
        # AI RESPONSE
        # ====================================================

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=messages,

            temperature=0.4,

            max_tokens=4000,

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

    # ========================================================
    # SAVE CHAT HISTORY
    # ========================================================

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
        k=8
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
1. Answer step-by-step
2. Mention trends
3. Mention anomalies
4. Mention correlations
5. Mention missing values
6. Give recommendations
7. Never hallucinate
8. Use only provided data

"""

    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

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

            temperature=0.2,

            max_tokens=3000,

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
# ============================================================
# INDUSTRIAL AI WORKSPACE
# ADVANCED AI ANALYTICS VERSION
# PREMIUM LIGHT EMERALD GLASS UI
# FULL FINAL ENTERPRISE VERSION
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
PREMIUM BUTTONS
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

.stButton button:hover {

    transform: translateY(-3px);

    border: 1px solid rgba(0,180,110,0.45);

    box-shadow:
    0 12px 30px rgba(0,180,110,0.16),
    0 0 24px rgba(0,180,110,0.18);
}

/* =========================================================
CHAT ANIMATION
========================================================= */

@keyframes smoothFadeIn {

    from {

        opacity: 0;

        transform: translateY(18px);
    }

    to {

        opacity: 1;

        transform: translateY(0px);
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
    0 8px 24px rgba(16,59,44,0.06);

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
    0 0 20px rgba(0,180,110,0.08);
}

.stChatInput input {

    background: transparent !important;

    color: #103B2C !important;

    font-size: 17px !important;

    font-weight: 500;
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
# ADVANCED DATA ANALYSIS ENGINE
# ============================================================

def advanced_dataframe_analysis(df):

    analysis = {}

    analysis["rows"] = df.shape[0]
    analysis["columns"] = df.shape[1]

    # Missing values

    missing = df.isnull().sum()

    analysis["missing_values"] = (
        missing[missing > 0].to_dict()
    )

    numeric_df = df.select_dtypes(include='number')

    if not numeric_df.empty:

        # Summary

        analysis["summary_statistics"] = (
            numeric_df.describe().to_dict()
        )

        # Trends

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

        # Anomalies

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

        # Correlations

        try:

            corr = numeric_df.corr()

            strong_corr = []

            for col1 in corr.columns:

                for col2 in corr.columns:

                    if col1 != col2:

                        value = corr.loc[col1, col2]

                        if abs(value) > 0.7:

                            strong_corr.append(
                                f"{col1} ↔ {col2} = {value:.2f}"
                            )

            analysis["strong_correlations"] = strong_corr

        except:
            analysis["strong_correlations"] = []

    return analysis

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">
    ⚡ Industrial AI
    </div>
    """, unsafe_allow_html=True)

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
# FILE UPLOAD
# ============================================================

if chat_mode == "📂 Upload & Analyze Reports":

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
                                title="Industrial Trend Analysis"
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                        analysis = advanced_dataframe_analysis(df)

                        text = f"""
DATASET:
{df.to_string(index=False)}

ADVANCED ANALYSIS:
{analysis}
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

                    analysis = advanced_dataframe_analysis(df)

                    text = f"""
DATASET:
{df.to_string(index=False)}

ADVANCED ANALYSIS:
{analysis}
"""

                    all_documents.append(
                        Document(page_content=text)
                    )

            # ========================================================
            # SPLITTING
            # ========================================================

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=250
            )

            chunks = splitter.split_documents(all_documents)

            # ========================================================
            # EMBEDDINGS
            # ========================================================

            @st.cache_resource
            def load_embeddings():

                return HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-en-v1.5"
                )

            embeddings = load_embeddings()

            # ========================================================
            # VECTOR STORE
            # ========================================================

            st.session_state.vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

            st.success("✅ Advanced analysis engine ready!")

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
    "Ask anything about industrial reports..."
)

# ============================================================
# GENERAL CHAT
# ============================================================

if (
    question
    and chat_mode == "🧠 General AI Chat"
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
            temperature=0.7,
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
# ADVANCED RAG CHATBOT
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

Your responsibilities:
- deeply analyze industrial datasets
- identify anomalies
- detect trends
- explain statistics
- identify operational risks
- identify KPI movement
- explain correlations
- provide recommendations
- avoid hallucination
- answer ONLY from context

ANALYSIS RULES:
1. Mention trends
2. Mention anomalies
3. Mention strong correlations
4. Mention missing values
5. Explain operational impact
6. Give recommendations
7. Be precise and data-driven
8. Never guess missing information

DATA CONTEXT:
{context}

USER QUESTION:
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
            temperature=0.2,
            max_tokens=2000,
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
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

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Industrial AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# PREMIUM CSS
# -----------------------
st.markdown("""
<style>

/* =========================
   GOOGLE FONT
========================= */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* =========================
   GLOBAL
========================= */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* =========================
   MAIN APP
========================= */
.stApp {

    background:
        radial-gradient(circle at top left, #1E1B4B 0%, transparent 25%),
        radial-gradient(circle at top right, #312E81 0%, transparent 25%),
        linear-gradient(180deg, #050816 0%, #0B1023 100%);

    color: #F8FAFC;
}

/* =========================
   MAIN CONTAINER
========================= */
.block-container {

    max-width: 1350px;

    padding-top: 1.2rem;

    padding-left: 3rem;

    padding-right: 3rem;

    padding-bottom: 2rem;
}

/* =========================
   SIDEBAR
========================= */
[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            rgba(10,15,30,0.98),
            rgba(5,8,22,0.98)
        );

    border-right:
        1px solid rgba(255,255,255,0.06);

    backdrop-filter: blur(18px);

    box-shadow:
        4px 0px 30px rgba(0,0,0,0.35);
}

[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

/* =========================
   HERO TITLE
========================= */
h1 {

    font-size: 4rem !important;

    font-weight: 800 !important;

    letter-spacing: -2px;

    margin-bottom: 0.5rem;

    background:
        linear-gradient(
            90deg,
            #A855F7,
            #6366F1,
            #38BDF8
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

/* =========================
   TEXT
========================= */
p, label, span, div {
    color: #E2E8F0;
}

/* =========================
   GLASS CARDS
========================= */
.kpi-card {

    background:
        rgba(255,255,255,0.05);

    backdrop-filter:
        blur(16px);

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius:
        24px;

    padding:
        26px;

    transition:
        all 0.3s ease;

    box-shadow:
        0px 6px 24px rgba(0,0,0,0.18);
}

.kpi-card:hover {

    transform:
        translateY(-6px);

    border:
        1px solid rgba(168,85,247,0.45);

    box-shadow:
        0px 10px 35px rgba(168,85,247,0.25);
}

/* =========================
   CHAT MESSAGES
========================= */
[data-testid="stChatMessage"] {

    background:
        rgba(255,255,255,0.05);

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius:
        24px;

    padding:
        18px;

    margin-bottom:
        18px;

    backdrop-filter:
        blur(14px);

    box-shadow:
        0px 6px 22px rgba(0,0,0,0.16);

    animation:
        fadeIn 0.3s ease-in-out;
}

/* USER MESSAGE */
[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {

    background:
        linear-gradient(
            90deg,
            rgba(99,102,241,0.20),
            rgba(168,85,247,0.18)
        );
}

/* ASSISTANT MESSAGE */
[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {

    background:
        rgba(255,255,255,0.04);
}

/* =========================
   CHAT INPUT
========================= */
.stChatInput {

    position:
        sticky;

    bottom:
        8px;

    padding-top:
        12px;

    background:
        transparent;
}

.stChatInput > div {

    background:
        rgba(15,23,42,0.88);

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius:
        24px;

    padding:
        10px;

    backdrop-filter:
        blur(16px);

    box-shadow:
        0px 8px 28px rgba(0,0,0,0.28);
}

.stChatInput input {

    background:
        transparent !important;

    border:
        none !important;

    color:
        white !important;

    font-size:
        16px !important;
}

/* =========================
   FILE UPLOADER
========================= */
[data-testid="stFileUploader"] {

    background:
        rgba(255,255,255,0.04);

    border:
        1px dashed rgba(168,85,247,0.45);

    border-radius:
        24px;

    padding:
        30px;

    backdrop-filter:
        blur(16px);

    transition:
        all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {

    border:
        1px dashed #A855F7;

    background:
        rgba(168,85,247,0.08);
}

/* =========================
   BUTTONS
========================= */
.stButton button {

    background:
        linear-gradient(
            90deg,
            #7C3AED,
            #6366F1
        );

    border:
        none;

    border-radius:
        16px;

    padding:
        12px 22px;

    font-weight:
        600;

    color:
        white !important;

    transition:
        all 0.3s ease;

    box-shadow:
        0px 4px 18px rgba(124,58,237,0.28);
}

.stButton button:hover {

    transform:
        translateY(-3px);

    box-shadow:
        0px 8px 28px rgba(124,58,237,0.45);
}

/* =========================
   RADIO BUTTONS
========================= */
.stRadio > div {

    background:
        rgba(255,255,255,0.04);

    border:
        1px solid rgba(255,255,255,0.06);

    padding:
        14px;

    border-radius:
        20px;
}

/* =========================
   ALERTS
========================= */
.stAlert {

    border-radius:
        18px !important;

    border:
        none !important;

    backdrop-filter:
        blur(14px);
}

/* =========================
   DATAFRAME
========================= */
[data-testid="stDataFrame"] {

    border-radius:
        20px;

    overflow:
        hidden;

    border:
        1px solid rgba(255,255,255,0.08);
}

/* =========================
   PLOTLY
========================= */
.js-plotly-plot {

    border-radius:
        24px;

    overflow:
        hidden;
}

/* =========================
   SCROLLBAR
========================= */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #0B1023;
}

::-webkit-scrollbar-thumb {

    background:
        linear-gradient(
            180deg,
            #7C3AED,
            #38BDF8
        );

    border-radius:
        12px;
}

/* =========================
   REMOVE STREAMLIT BRANDING
========================= */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* =========================
   ANIMATION
========================= */
@keyframes fadeIn {

    from {
        opacity: 0;
        transform: translateY(10px);
    }

    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

/* =========================
   MOBILE RESPONSIVE
========================= */
@media (max-width: 768px) {

    .block-container {

        padding-left: 1rem !important;

        padding-right: 1rem !important;
    }

    h1 {

        font-size: 2.5rem !important;
    }

    .kpi-card {

        margin-bottom: 16px;
    }

    [data-testid="stSidebar"] {

        width: 250px !important;
    }
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# SESSION STATE
# -----------------------
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
    st.session_state.chat_mode = "🧠 General AI Chat"

# -----------------------
# SIDEBAR
# -----------------------
with st.sidebar:

    st.markdown("# ⚡ Industrial")
    st.markdown("### Your AI Co-pilot")

    st.markdown("---")

    st.markdown("## 📊 Dashboard")
    st.markdown("## 📁 Files")
    st.markdown("## 🤖 AI Insights")
    st.markdown("## 📈 Analytics")

    st.markdown("---")

    if st.button("🔄 New Analysis"):

        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.session_state.files_processed = False
        st.session_state.report_count = 0
        st.session_state.uploaded_files = []

        st.rerun()

# -----------------------
# HERO SECTION
# -----------------------
st.markdown("""
<h1>
⚡ Industrial AI Assistant
</h1>
""", unsafe_allow_html=True)

st.markdown("""
### AI-powered Industrial Analytics & RAG Platform
""")

# -----------------------
# GREETING
# -----------------------
st.info("""
👋 Hello! Welcome to Industrial AI Assistant.

Choose what you want to do today:
""")

# -----------------------
# MODE SELECTION
# -----------------------
chat_mode = st.radio(
    "Select AI Mode",
    [
        "🧠 General AI Chat",
        "📂 Upload & Analyze Reports"
    ],
    horizontal=True
)

st.session_state.chat_mode = chat_mode

# -----------------------
# MODE MESSAGE
# -----------------------
if st.session_state.chat_mode == "🧠 General AI Chat":

    st.success(
        "You are now using General AI Chat Mode."
    )

else:

    st.success(
        "Upload industrial reports to activate analytics and RAG intelligence."
    )

# -----------------------
# KPI VALUES
# -----------------------
report_count = st.session_state.report_count
query_count = len(st.session_state.chat_history)

if query_count == 0:
    efficiency = 100

elif query_count <= 5:
    efficiency = 95 + query_count

else:
    efficiency = 99

status = "Active" if st.session_state.files_processed else "Waiting"

# -----------------------
# KPI CARDS
# -----------------------
col1, col2, col3, col4 = st.columns(4)

cards = [
    ("📄 Reports", str(report_count), "#FFFFFF"),
    ("🤖 AI Queries", str(query_count), "#FFFFFF"),
    ("⚡ Efficiency", f"{efficiency}%", "#10B981"),
    ("📈 Status", status, "#38BDF8")
]

for col, card in zip([col1, col2, col3, col4], cards):

    with col:

        st.markdown(f"""
        <div class="kpi-card">
        <h4 style="color:#CBD5E1;">{card[0]}</h4>
        <h2 style="color:{card[2]};">{card[1]}</h2>
        </div>
        """, unsafe_allow_html=True)

# -----------------------
# FILE UPLOADER
# -----------------------
if (
    st.session_state.chat_mode
    == "📂 Upload & Analyze Reports"
):

    st.markdown("## 📂 Upload Industrial Reports")

    uploaded_files = st.file_uploader(
        "Upload Reports",
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

# -----------------------
# ANALYTICS DASHBOARD
# -----------------------
if (
    st.session_state.files_processed
    and st.session_state.chat_mode
    == "📂 Upload & Analyze Reports"
):

    st.markdown("## 📊 Analytics Dashboard")

    try:

        uploaded_files = st.session_state.uploaded_files

        if uploaded_files:

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

                numeric_cols = df.select_dtypes(
                    include='number'
                ).columns

                if len(numeric_cols) > 0:

                    target_col = numeric_cols[0]

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric(
                            "Average",
                            round(df[target_col].mean(), 2)
                        )

                    with c2:
                        st.metric(
                            "Maximum",
                            round(df[target_col].max(), 2)
                        )

                    with c3:
                        st.metric(
                            "Minimum",
                            round(df[target_col].min(), 2)
                        )

                    fig1 = px.line(
                        df,
                        y=target_col,
                        title=f"{target_col} Trend"
                    )

                    st.plotly_chart(
                        fig1,
                        use_container_width=True
                    )

    except Exception as e:

        st.error(f"Analytics unavailable: {e}")

# -----------------------
# DISPLAY CHAT
# -----------------------
for chat in st.session_state.chat_history:

    with st.chat_message("user", avatar="👨‍💻"):
        st.write(chat["question"])

    with st.chat_message("assistant", avatar="🤖"):
        st.write(chat["answer"])

# -----------------------
# CHAT INPUT
# -----------------------
question = st.chat_input(
    "Ask your question..."
)

# -----------------------
# GENERAL AI CHAT
# -----------------------
if (
    question
    and st.session_state.chat_mode
    == "🧠 General AI Chat"
):

    with st.chat_message("user", avatar="👨‍💻"):
        st.write(question)

    api_key = st.secrets["GROQ_API_KEY"]

    client = Groq(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": """
You are a highly intelligent, friendly, and conversational AI assistant.

Speak naturally like ChatGPT.
Be smooth, modern, engaging, and helpful.
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

    with st.chat_message("assistant", avatar="🤖"):

        message_placeholder = st.empty()

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

                message_placeholder.markdown(
                    full_response + "▌"
                )

        message_placeholder.markdown(full_response)

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": full_response
        }
    )

# -----------------------
# RAG AI RESPONSE
# -----------------------
if (
    question
    and st.session_state.vectorstore is not None
    and st.session_state.chat_mode
    == "📂 Upload & Analyze Reports"
):

    with st.chat_message("user", avatar="👨‍💻"):
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

    with st.chat_message("assistant", avatar="🤖"):

        message_placeholder = st.empty()

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

                message_placeholder.markdown(
                    full_response + "▌"
                )

        message_placeholder.markdown(full_response)

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": full_response
        }
    )
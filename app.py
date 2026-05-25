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
    layout="wide"
)

# -----------------------
# PREMIUM CSS
# -----------------------
st.markdown("""
<style>

.stApp {
    background: #F4F7FE;
    color: #111827;
}

.block-container {
    padding-top: 1.5rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 1400px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#050816,#0B1023);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

h1 {
    font-size: 3.5rem !important;
    font-weight: 800 !important;
}

[data-testid="stChatMessage"] {
    background: white;
    border-radius: 22px;
    padding: 20px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 18px;
}

.stButton button {
    background: linear-gradient(90deg,#6C63FF,#8B5CF6);
    color: white !important;
    border-radius: 14px;
    border: none;
    padding: 12px 20px;
    font-weight: 600;
}

.kpi-card {
    background: white;
    padding: 22px;
    border-radius: 22px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
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

    st.markdown("# ⚡ Industrial AI")
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
<h1 style="
background: linear-gradient(90deg,#6C63FF,#8B5CF6);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
">
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
    ("📄 Reports", str(report_count), "#111827"),
    ("🤖 AI Queries", str(query_count), "#111827"),
    ("⚡ Efficiency", f"{efficiency}%", "#10B981"),
    ("📈 Status", status, "#6366F1")
]

for col, card in zip([col1, col2, col3, col4], cards):

    with col:

        st.markdown(f"""
        <div class="kpi-card">
        <h4 style="color:#6B7280;">{card[0]}</h4>
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

    st.markdown("""
    ### 📂 Upload Industrial Reports
    """)

    uploaded_files = st.file_uploader(
        "Upload Reports",
        type=["pdf", "xlsx", "csv"],
        accept_multiple_files=True
    )

    # -----------------------
    # PROCESS FILES
    # -----------------------
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

                # PDF
                if uploaded_file.name.endswith(".pdf"):

                    loader = PyPDFLoader(temp_path)

                    documents = loader.load()

                    all_documents.extend(documents)

                # EXCEL
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

                # CSV
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

            # -----------------------
            # REAL EMBEDDINGS
            # -----------------------
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

                # CSV
                if uploaded_file.name.endswith(".csv"):

                    df = pd.read_csv(uploaded_file)

                # EXCEL
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

                    # LINE CHART
                    fig1 = px.line(
                        df,
                        y=target_col,
                        title=f"{target_col} Trend"
                    )

                    st.plotly_chart(
                        fig1,
                        use_container_width=True
                    )

                    # BAR CHART
                    fig2 = px.bar(
                        df,
                        y=target_col,
                        title=f"{target_col} Comparison"
                    )

                    st.plotly_chart(
                        fig2,
                        use_container_width=True
                    )

                    # PIE CHART
                    try:

                        fig3 = px.pie(
                            df,
                            names=df.columns[0],
                            values=target_col,
                            title=f"{target_col} Distribution"
                        )

                        st.plotly_chart(
                            fig3,
                            use_container_width=True
                        )

                    except:
                        pass

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

    # -----------------------
    # CONVERSATIONAL MEMORY
    # -----------------------
    messages = [
        {
            "role": "system",
            "content": """
You are a highly intelligent, friendly, and conversational AI assistant.

Your personality is:
- smooth
- modern
- engaging
- professional
- helpful

You speak naturally like ChatGPT.

You can:
- answer general questions
- explain AI concepts
- help with coding
- discuss industrial engineering
- explain analytics
- solve technical doubts
- guide users professionally

Rules:
- Keep conversations natural
- Avoid robotic replies
- Be engaging and intelligent
- Explain clearly
- Use examples when useful
- Maintain conversational flow
- Talk like a modern AI assistant
"""
        }
    ]

    # ADD PREVIOUS CHAT HISTORY
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

    # CURRENT QUESTION
    messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # -----------------------
    # STREAMING RESPONSE
    # -----------------------
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

    # SAVE CHAT HISTORY
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

Your personality is:
- conversational
- intelligent
- professional
- smooth

Rules:
- Give detailed explanations
- Mention trends
- Give recommendations
- Be natural and engaging
- Explain clearly
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
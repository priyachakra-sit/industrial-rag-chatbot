import pandas as pd
import plotly.express as px
from langchain_core.documents import Document
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FakeEmbeddings
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

/* APP */
.stApp {
    background: #F4F7FE;
    color: #111827;
}

/* MAIN CONTAINER */
.block-container {
    padding-top: 1.5rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 1400px;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#050816,#0B1023);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* HERO TITLE */
h1 {
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    line-height: 1.1;
}

/* CHAT MESSAGES */
[data-testid="stChatMessage"] {
    background: white;
    border-radius: 22px;
    padding: 20px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 18px;
    color: #111827 !important;
}

/* FORCE CHAT TEXT COLOR */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: #111827 !important;
}

/* USER CHAT */
[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(90deg,#EEF2FF,#F5F3FF);
}

/* AI CHAT */
[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    background: white;
}

/* CHAT INPUT */
.stChatInput {
    position: sticky;
    bottom: 0;
    background: transparent;
    padding-top: 10px;
}

.stChatInput input {
    border-radius: 18px !important;
    border: 1px solid #E5E7EB !important;
    padding: 18px !important;
    background: white !important;
    font-size: 16px !important;
    color: #111827 !important;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 22px;
    padding: 22px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
}

/* BUTTONS */
.stButton button {
    background: linear-gradient(90deg,#6C63FF,#8B5CF6);
    color: white !important;
    border-radius: 14px;
    border: none;
    padding: 12px 20px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0px 6px 20px rgba(108,99,255,0.3);
}

/* SUCCESS ALERT */
.stAlert {
    border-radius: 18px;
}

/* KPI CARDS */
.kpi-card {
    background: white;
    padding: 22px;
    border-radius: 22px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #C7D2FE;
    border-radius: 10px;
}

/* REMOVE STREAMLIT STUFF */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* MOBILE RESPONSIVE */
@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1rem !important;
    }

    h1 {
        font-size: 2.2rem !important;
    }

    [data-testid="stSidebar"] {
        width: 250px !important;
    }

    .stChatInput input {
        font-size: 14px !important;
    }

    [data-testid="stChatMessage"] {
        padding: 16px;
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

# -----------------------
# SIDEBAR
# -----------------------
with st.sidebar:

    st.markdown("""
    # ⚡ Industrial AI

    ### Your AI Co-pilot
    """)

    st.markdown("---")

    st.markdown("## 📊 Dashboard")
    st.markdown("## 📁 Files")
    st.markdown("## 🤖 AI Insights")
    st.markdown("## 📈 Analytics")
    st.markdown("## ⚙ Settings")

    st.markdown("---")

    st.markdown("""
    ### 🚀 Features

    ✅ Multi-File Analysis  
    ✅ AI Insights  
    ✅ Smart Recommendations  
    ✅ Industrial Analytics  
    ✅ RAG Search  
    """)

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
<div style="padding-top:20px;">

<h1 style="
font-size:58px;
font-weight:800;
margin-bottom:0;
background: linear-gradient(90deg,#6C63FF,#8B5CF6);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
">
⚡ Industrial AI Assistant
</h1>

<p style="
font-size:22px;
color:#6B7280;
margin-top:10px;
">
Advanced AI Platform for Industrial Reports & Analytics
</p>

<p style="
font-size:16px;
color:#9CA3AF;
margin-top:5px;
">
Upload industrial reports and receive AI-powered insights, trends, analytics, and recommendations.
</p>

</div>
""", unsafe_allow_html=True)

# -----------------------
# DYNAMIC KPI VALUES
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
if not st.session_state.files_processed:

    st.markdown("""
    <div style="
    background:white;
    padding:25px;
    border-radius:24px;
    box-shadow:0 4px 20px rgba(0,0,0,0.05);
    margin-top:25px;
    margin-bottom:25px;
    ">
    <h3>📂 Upload Industrial Reports</h3>
    <p style="color:#6B7280;">
    Supported formats: PDF, Excel, CSV
    </p>
    </div>
    """, unsafe_allow_html=True)

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

            @st.cache_resource
            def load_embeddings():
                return FakeEmbeddings(size=384)

            embeddings = load_embeddings()

            st.session_state.vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

            st.session_state.files_processed = True

            st.success(
                f"✅ {len(uploaded_files)} file(s) processed successfully!"
            )

            st.rerun()

        except Exception as e:

            st.error(f"Error processing file: {e}")

# -----------------------
# ANALYTICS DASHBOARD
# -----------------------
if st.session_state.files_processed:

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

                else:

                    st.warning(
                        "No numeric columns available for analytics."
                    )

    except Exception as e:

        st.error(f"Analytics unavailable: {e}")

# -----------------------
# FILE STATUS
# -----------------------
if st.session_state.files_processed:

    st.markdown("""
    <div style="
    background:linear-gradient(90deg,#6C63FF,#8B5CF6);
    padding:16px;
    border-radius:18px;
    color:white;
    margin-top:20px;
    margin-bottom:20px;
    font-weight:600;
    ">
    ✅ Reports uploaded successfully. Ask unlimited questions about your files.
    </div>
    """, unsafe_allow_html=True)

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
    "Ask anything about your uploaded reports..."
)

# -----------------------
# AI RESPONSE
# -----------------------
if question and st.session_state.vectorstore is not None:

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

Your job is to deeply analyze uploaded reports and provide detailed, professional, and structured responses.

Rules:
- Give clear and detailed explanations
- Use headings and bullet points
- Mention exact values from data
- Explain trends and abnormalities
- Give industrial recommendations
- Compare values when needed
- Summarize findings professionally
- Answer ONLY from provided context
- Keep responses detailed and insightful

CONTEXT:
{context}

QUESTION:
{question}
"""

    api_key = st.secrets["GROQ_API_KEY"]

    client = Groq(api_key=api_key)

    with st.spinner("🤖 AI is analyzing industrial reports..."):

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=1800
        )

    answer = response.choices[0].message.content

    with st.chat_message("assistant", avatar="🤖"):
        st.write(answer)

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": answer
        }
    )
import pandas as pd
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
# CUSTOM UI
# -----------------------
st.markdown("""
<style>

/* MAIN APP */
.stApp {
    background: #F4F7FE;
    color: #111827;
}

/* MAIN CONTAINER */
.block-container {
    padding-top: 2rem;
    padding-left: 4rem;
    padding-right: 4rem;
    max-width: 1200px;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#050816,#0B1023);
    width: 320px !important;
    min-width: 320px !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] * {
    color: white;
}

/* TITLE */
h1 {
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    color: #5B5FEF !important;
    letter-spacing: -1px;
}

/* SUBTITLE */
h3 {
    color: #6B7280 !important;
    font-weight: 500 !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 20px;
    padding: 20px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
}

/* CHAT INPUT */
.stChatInput input {
    border-radius: 20px !important;
    border: 1px solid #E5E7EB !important;
    padding: 18px !important;
    background: white !important;
    font-size: 16px !important;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}

/* CHAT MESSAGE */
[data-testid="stChatMessage"] {
    background: white;
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 20px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.04);
}

/* SUCCESS BOX */
.stAlert {
    border-radius: 18px;
}

/* BUTTON */
.stButton button {
    background: linear-gradient(90deg,#6C63FF,#8B5CF6);
    color: white;
    border-radius: 14px;
    border: none;
    padding: 12px 20px;
    font-weight: 600;
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

</style>
""", unsafe_allow_html=True)


# -----------------------
# SESSION STATE
# -----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

# -----------------------
# TITLE
# -----------------------

st.markdown("""
<div style="padding-top:20px;">

<h1 style="
font-size:55px;
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

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="
    background:white;
    padding:20px;
    border-radius:20px;
    box-shadow:0 4px 15px rgba(0,0,0,0.05);
    ">
    <h4 style="color:#6B7280;">📄 Reports</h4>
    <h2 style="color:#111827;">12</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="
    background:white;
    padding:20px;
    border-radius:20px;
    box-shadow:0 4px 15px rgba(0,0,0,0.05);
    ">
    <h4 style="color:#6B7280;">🤖 AI Queries</h4>
    <h2 style="color:#111827;">58</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="
    background:white;
    padding:20px;
    border-radius:20px;
    box-shadow:0 4px 15px rgba(0,0,0,0.05);
    ">
    <h4 style="color:#6B7280;">⚡ Efficiency</h4>
    <h2 style="color:#10B981;">94%</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="
    background:white;
    padding:20px;
    border-radius:20px;
    box-shadow:0 4px 15px rgba(0,0,0,0.05);
    ">
    <h4 style="color:#6B7280;">📈 Status</h4>
    <h2 style="color:#6366F1;">Active</h2>
    </div>
    """, unsafe_allow_html=True)


# -----------------------
# API KEY
# -----------------------
api_key = st.secrets["GROQ_API_KEY"]

# -----------------------
# FILE UPLOADER
# -----------------------
st.markdown("""
<div style="
background:white;
padding:25px;
border-radius:24px;
box-shadow:0 4px 20px rgba(0,0,0,0.05);
margin-top:25px;
margin-bottom:25px;
">
<h3 style="margin-bottom:15px;">📂 Upload Industrial Reports</h3>
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
# MAIN PROCESS
# -----------------------
if uploaded_files:

    all_documents = []

    # -----------------------
    # PROCESS FILES
    # -----------------------
    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{uploaded_file.name.split('.')[-1]}"
        ) as tmp:

            tmp.write(uploaded_file.read())

            temp_path = tmp.name

        # -----------------------
        # PDF
        # -----------------------
        if uploaded_file.name.endswith(".pdf"):

            loader = PyPDFLoader(temp_path)

            documents = loader.load()

            all_documents.extend(documents)

        # -----------------------
        # EXCEL
        # -----------------------
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

        # -----------------------
        # CSV
        # -----------------------
        elif uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(temp_path)

            text = df.to_string(index=False)

            all_documents.append(
                Document(page_content=text)
            )

    st.success(f"{len(uploaded_files)} file(s) processed successfully!")

    # -----------------------
    # TEXT SPLITTING
    # -----------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_documents)

    # -----------------------
    # EMBEDDINGS
    # -----------------------
    embeddings = FakeEmbeddings(size=384)

    # -----------------------
    # VECTOR DATABASE
    # -----------------------
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # -----------------------
    # QUESTION INPUT
    # -----------------------
    question = st.chat_input("Ask your question")

    if question:

        # -----------------------
        # USER MESSAGE
        # -----------------------
        with st.chat_message("user"):
            st.write(question)

        # -----------------------
        # RETRIEVAL
        # -----------------------
        docs = vectorstore.similarity_search(
            question,
            k=5
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # -----------------------
        # PROMPT
        # -----------------------
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

        # -----------------------
        # GROQ CLIENT
        # -----------------------
        client = Groq(api_key=api_key)

        # -----------------------
        # LLM RESPONSE
        # -----------------------
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
                max_tokens=3000
            )


        answer = response.choices[0].message.content

        # -----------------------
        # SAVE CHAT
        # -----------------------
        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )

# -----------------------
# DISPLAY CHAT
# -----------------------
for chat in st.session_state.chat_history:

    with st.chat_message("user", avatar="👨‍💻"):
        st.write(chat["question"])

    with st.chat_message("assistant", avatar="🤖"):
        st.write(chat["answer"])
        
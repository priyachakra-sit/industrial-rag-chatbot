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

/* MAIN BACKGROUND */
.stApp {
    background-color: #F5F7FB;
}

/* MAIN CONTENT */
.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050816, #0B1023);
    border-right: 1px solid #1E293B;
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] * {
    color: white;
}

/* TITLE */
h1 {
    font-size: 3rem !important;
    font-weight: 700 !important;
    color: #5B5FEF !important;
}

/* CHAT INPUT */
.stChatInput input {
    border-radius: 15px !important;
    border: 1px solid #D1D5DB !important;
    padding: 14px !important;
    background-color: white !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background-color: white;
    border-radius: 15px;
    padding: 15px;
    border: 1px solid #E5E7EB;
}

/* SUCCESS BOX */
.stAlert {
    border-radius: 15px;
}

/* CHAT MESSAGE */
[data-testid="stChatMessage"] {
    background-color: white;
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 15px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}

/* BUTTONS */
.stButton button {
    border-radius: 12px !important;
    background: linear-gradient(90deg,#6C63FF,#8B5CF6);
    color: white !important;
    border: none !important;
    padding: 10px 18px;
    font-weight: 600;
}

/* REMOVE STREAMLIT MENU */
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

    st.markdown("# ⚡ Industrial AI")

    st.markdown("### Your AI Copilot")

    st.divider()

    st.markdown("## 📊 Dashboard")

    st.markdown("## 📁 Files")

    st.markdown("## 🤖 AI Insights")

    st.markdown("## 📈 Analytics")

    st.markdown("## ⚙ Settings")

    st.divider()

    st.markdown("""
    ### 🚀 Features

    ✅ Multi-file Analysis  
    ✅ AI Insights  
    ✅ RAG Search  
    ✅ Industrial Analytics  
    ✅ Smart Recommendations  
    """)

# -----------------------
# TITLE
# -----------------------
st.markdown("# ⚡ Industrial AI Assistant")

st.markdown("""
### Advanced AI Assistant for Industrial Reports
""")

st.caption("Advanced AI Assistant for Industrial Reports")

st.markdown("""
Upload industrial reports and receive AI-powered insights, analysis, trends, and recommendations.
""")

# -----------------------
# API KEY
# -----------------------
api_key = st.secrets["GROQ_API_KEY"]

# -----------------------
# FILE UPLOADER
# -----------------------
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

    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])
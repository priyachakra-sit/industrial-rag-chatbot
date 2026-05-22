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
# SESSION STATE
# -----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Industrial AI Assistant")

# -----------------------
# SIDEBAR
# -----------------------
with st.sidebar:
    st.header("About")

    st.write("""
    Upload industrial reports and ask questions.

    Supported:
    - PDF
    - Excel
    - CSV

    Powered by:
    - Streamlit
    - FAISS
    - Groq LLM
    """)

# -----------------------
# TITLE
# -----------------------
st.title("⚡ Industrial AI Assistant")

st.caption("Assistant for Industrial Reports")

st.write("Upload your industrial report and ask questions.")

# -----------------------
# API KEY
# -----------------------
api_key = st.secrets["GROQ_API_KEY"]

# -----------------------
# FILE UPLOAD
# -----------------------
uploaded_file = st.file_uploader(
    "Upload Report",
    type=["pdf", "xlsx", "csv"]
)

# -----------------------
# MAIN PROCESS
# -----------------------
if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{uploaded_file.name.split('.')[-1]}"
    ) as tmp:

        tmp.write(uploaded_file.read())

        temp_path = tmp.name

    # -----------------------
    # LOAD FILE
    # -----------------------
    if uploaded_file.name.endswith(".pdf"):

        loader = PyPDFLoader(temp_path)

        documents = loader.load()

    elif uploaded_file.name.endswith(".xlsx"):

        excel_data = pd.read_excel(
            temp_path,
            sheet_name=None
        )

        text = ""

        for sheet_name, df in excel_data.items():

            text += f"\n\nSheet: {sheet_name}\n"

            text += df.to_string(index=False)

        documents = [
            Document(page_content=text)
        ]

    elif uploaded_file.name.endswith(".csv"):

        df = pd.read_csv(temp_path)

        text = df.to_string(index=False)

        documents = [
            Document(page_content=text)
        ]

    # -----------------------
    # TEXT SPLITTING
    # -----------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

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

    st.success(f"{uploaded_file.name} processed successfully!")

    # -----------------------
    # USER QUESTION
    # -----------------------
    question = st.text_input("Ask your question")

    if question:

        # -----------------------
        # RETRIEVAL
        # -----------------------
        docs = vectorstore.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # -----------------------
        # PROMPT
        # -----------------------
        prompt = f"""
You are an industrial energy audit assistant.

Answer ONLY from the provided context.

Rules:
- Give concise answers
- Do not show unnecessary calculations
- Give final results clearly
- Use bullet points when needed
- Mention exact values from data

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
            max_tokens=1000
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
# DISPLAY CHAT HISTORY
# -----------------------
for chat in st.session_state.chat_history:

    st.markdown("### Question:")
    st.write(chat["question"])

    st.markdown("### Answer:")
    st.write(chat["answer"])

    st.divider()
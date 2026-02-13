import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import requests
import tempfile

st.set_page_config(page_title="Smart Document Q&A", layout="wide")
st.title("📄 Smart Document Q&A System (Cloud Version)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

@st.cache_resource
def process_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def query_huggingface(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error: {response.text}"


if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    vectorstore = process_document(file_path)
    st.success("Document processed successfully!")

    question = st.text_input("Ask a question about your document")

    if question:
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
        Answer the question using only the context below.

        Context:
        {context}

        Question:
        {question}
        """

        with st.spinner("Generating answer..."):
            answer = query_huggingface(prompt)

        st.subheader("Answer")
        st.write(answer)

import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Smart Document Q&A", layout="wide")
st.title("📄 Smart Document Q&A System (Cloud Version)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# ---------------- DOCUMENT PROCESSING ----------------
@st.cache_resource
def process_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    vectorstore = process_document(tmp_path)

    st.success("Document processed successfully!")

    question = st.text_input("Ask a question about your document")

    if question:

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(question)

        context = "\n\n".join([doc.page_content for doc in docs])

        # 🔥 NEW HuggingFace Endpoint (Router Compatible)
        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-base",
            temperature=0,
            max_new_tokens=512,
        )

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question using ONLY the context below.
            If the answer is not found, say "Not found in document".

            Context:
            {context}

            Question:
            {question}
            """
        )

        final_prompt = prompt.format(context=context, question=question)

        response = llm.invoke(final_prompt)

        st.subheader("Answer")
        st.write(response)

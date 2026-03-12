# Smart Document AI 📄🤖

An AI-powered RAG (Retrieval Augmented Generation) application that allows users to upload PDF documents and ask questions directly from them.

The system analyzes the document and generates accurate answers using semantic search and AI models.

---

## 🚀 Features

• Multi-PDF upload  
• ChatGPT-style interface  
• Resume scoring (1–10)  
• ATS match analysis  
• Hybrid retrieval (semantic + keyword search)  
• Fast document question answering  
• Works with large PDFs  

---

## 💡 Why This Project?

Large PDFs can contain hundreds of pages and it becomes difficult to find specific information quickly.

Instead of scrolling through the entire document, users can simply:

1. Upload the PDF
2. Ask a question
3. Get the answer instantly

Example:

"Find my CGPA from this resume"

---

## 🧠 Technologies Used

• Python  
• Streamlit  
• FAISS (Vector Database)  
• Sentence Transformers  
• HuggingFace LLM API  
• PyPDF2  

---

## ⚙️ How It Works

1️⃣ User uploads PDF  
2️⃣ Text is extracted from the document  
3️⃣ Text is split into chunks  
4️⃣ Embeddings are created  
5️⃣ Stored inside FAISS vector database  
6️⃣ User question is converted into embedding  
7️⃣ Relevant context is retrieved  
8️⃣ AI generates the final answer  

---

## 🖥️ Demo

Upload your PDF and ask questions like:

• "What is the CGPA?"  
• "Summarize this document"  
• "Explain the project in points"

---

## 📂 Project Structure

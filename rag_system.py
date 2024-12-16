# rag_system.py
import os
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import SimpleDirectoryReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import gradio as gr

# Load LLaMA 2 Model from Hugging Face
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Change this to the desired open-source model

print("Loading LLaMA 2 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Load and Split Documents
data_dir = "./data"
loader = SimpleDirectoryReader(data_dir)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Generate Embeddings and Build FAISS Index
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # Efficient for smaller datasets
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("./faiss_index")

# Load the FAISS index
vector_store = FAISS.load_local("./faiss_index", embeddings)
retriever = vector_store.as_retriever()

# Define the Q&A system using the loaded LLaMA 2 model
def qa_system(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs[:3]])  # Combine top 3 documents
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=500)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Gradio Interface
gr.Interface(fn=qa_system, inputs="text", outputs="text", title="Research Paper Q&A System with LLaMA 2").launch()

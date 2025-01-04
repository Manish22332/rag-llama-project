# rag_system.py
import os
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import util  # For cosine similarity
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
# Initialize history as an empty list
history = []

def qa_system(query):
    global history
    
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(query)
    doc_context = "\n".join([doc.page_content for doc in docs[:2]])

    # Concatenate the last few conversation turns to maintain context
    context = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history[-3:]])
    input_text = f"Document Context: {doc_context}\n{context}\nUser: {query}\nAssistant:"

    # Generate multiple responses
    inputs = tokenizer(input_text, return_tensors="pt")
    responses = model.generate(**inputs, num_return_sequences=2, max_length=500)

    # Convert responses to text
    response_texts = [tokenizer.decode(r, skip_special_tokens=True) for r in responses]

    # Embed the query and responses for ranking
    query_embedding = embeddings.embed_query(query)
    scores = [util.cos_sim(query_embedding, embeddings.embed_query(resp))[0][0].item() for resp in response_texts]

    # Select the best response based on the highest similarity score
    best_response = response_texts[scores.index(max(scores))]

    # Add current query and best response to history
    history.append((query, best_response))
    return best_response

# Gradio Interface
gr.Interface(fn=qa_system, inputs="text", outputs="text", title="Research Paper Q&A System with LLaMA 2").launch()

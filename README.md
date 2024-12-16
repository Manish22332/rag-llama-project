Research Paper Q&A System with LLaMA 2

Project Overview

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) system using LLaMA 2 for answering questions based on research papers. The system integrates:

FAISS for vector search

Hugging Face Transformers for both embeddings and response generation

LangChain for document processing and retrieval logic

Gradio for a web-based interface

The goal is to retrieve the most relevant sections from research papers and generate accurate answers using LLaMA 2.

Project Structure

rag_project/
  ├── data/                  # Folder for research papers
  ├── faiss_index/           # Folder for the FAISS index
  ├── rag_system.py          # Main Python script
  ├── requirements.txt       # Python dependencies
  └── README.md              # Project documentation

Setup and Installation

1. Prerequisites

Python 3.8+

Install dependencies from requirements.txt:

pip install -r requirements.txt

2. Download LLaMA 2 Model

Ensure you have access to the LLaMA 2 model through Hugging Face. You can modify the MODEL_NAME in the script to point to your desired model.

3. Prepare Data

Place your research papers in the ./data directory as PDF files.

4. Run the System

python rag_system.py

This will launch a Gradio interface where you can enter your question and get an answer based on the most relevant documents.
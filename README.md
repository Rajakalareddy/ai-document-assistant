# AI Document Assistant

An AI-powered PDF question-answering application built with Python and Streamlit.

## Features

- Upload PDF documents
- Extract text from PDFs
- Split documents into text chunks
- Generate embeddings using Hugging Face
- Store and search embeddings with FAISS
- Ask questions about uploaded documents
- Run locally without a paid OpenAI API

## Technologies Used

- Python
- Streamlit
- PyPDF
- LangChain
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- PyTorch

## How It Works

1. User uploads a PDF.
2. Text is extracted from the document.
3. The text is divided into smaller chunks.
4. Hugging Face embeddings are generated.
5. FAISS stores the embeddings for semantic search.
6. Relevant document content is retrieved based on the user's question.
7. A local question-answering model extracts an answer from the relevant context.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt

import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

load_dotenv()
@st.cache_resource
def load_local_model():
    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

tokenizer, qa_model = load_local_model()
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI Document Assistant")
st.write("Upload a PDF and ask questions about its content.")

uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type="pdf"
)

if uploaded_file:
    st.success("PDF uploaded successfully!")

    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    st.write(f"Extracted {len(text)} characters from the PDF.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    st.write(f"Created {len(chunks)} text chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )

    st.write("Vector database created successfully.")

    question = st.text_input("Ask a question about the PDF")

    if question:
        docs = vector_store.similarity_search(question, k=2)

        context = "\n\n".join([doc.page_content for doc in docs])

        inputs = tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = qa_model(**inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        best_score = float("-inf")
        answer_start = 0
        answer_end = 0

        for start in range(len(start_logits)):
            for end in range(start, min(start + 30, len(end_logits))):
                score = start_logits[start] + end_logits[end]
                if score > best_score:
                    best_score = score
                    answer_start = start
                    answer_end = end + 1

        answer = tokenizer.decode(
            inputs["input_ids"][0][answer_start:answer_end],
            skip_special_tokens=True
        )

        st.subheader("Answer")
        st.write(answer)
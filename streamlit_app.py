# esg_streamlit_app.py

import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub

# --- Configuration ---
VECTOR_STORE = "./vector_store/"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_TOKEN = st.secrets["HF_TOKEN"]  # set in Streamlit secrets

# --- Load Embedding and FAISS DBs ---
@st.cache_resource(show_spinner=False)
def load_vector_stores():
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.load_local(VECTOR_STORE, embedding, allow_dangerous_deserialization=True)
    
    return embedding, db

embedding, db = load_vector_stores()

# --- Prompt Template ---
prompt_template = PromptTemplate.from_template(
    """
You are an ESG analyst. Use only the context provided below to answer the question.

Context:
{context}

Question: {question}

Helpful Answer (focus only on the company in context, and do not hallucinate other companies):
"""
)

# --- Streamlit UI ---
st.set_page_config(page_title="ESG Profile Generator", layout="wide")
st.title("🌱 ESG Company Profile Generator")

company = st.text_input("Enter Company Name (e.g., TCS, INFY, HDFCBANK)")
custom_question = st.text_area("Optional: Custom Question", value="Generate ESG company profile")
k = st.slider("Number of documents to search", min_value=2, max_value=10, value=4)

if st.button("Generate ESG Profile") and company:
    with st.spinner("Retrieving relevant information and generating profile..."):
        docs = db.similarity_search(company, k=k)
        documents = docs

        context = "\n\n".join([doc.page_content for doc in documents])
        final_prompt = prompt_template.format(question=custom_question, context=context)

        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            model_kwargs={"temperature": 0.2, "max_new_tokens": 512},
            huggingfacehub_api_token=HF_API_TOKEN
        )

        response = llm.invoke(final_prompt)

        st.markdown("## 🧾 ESG Profile")
        st.markdown(response)

        st.markdown("---")
        st.markdown("## 📚 Sources")
        for i, doc in enumerate(documents):
            st.markdown(f"**{i+1}.** `{doc.metadata.get('source')}` | Page {doc.metadata.get('page')} | Year {doc.metadata.get('year')}")
            st.caption(doc.page_content[:300] + "...")

# streamlit_app.py

import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub  # Optional, can be replaced with any other LLM

# Set Hugging Face API Token if using hosted models
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["hf_token"]

# --- Configuration ---
VECTOR_STORE_PATH = os.path.join(".", "vector_store")  # Adjust if different
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Initialize components ---
@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

@st.cache_resource
def load_vectorstore():
    return FAISS.load_local(VECTOR_STORE_PATH, embedding, allow_dangerous_deserialization=True)

@st.cache_resource
def load_llm():
    # Replace this with any LLM you're allowed to use (OpenAI, HuggingFaceHub, etc.)
    return HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.3, "max_length": 512})

embedding = load_embedding()
vectorstore = load_vectorstore()
llm = load_llm()

# --- Prompt Template ---
prompt_template = PromptTemplate.from_template("""
You are an ESG analyst. Use only the context provided below to answer the question.

Context:
{context}

Question: {question}

Helpful Answer:
""")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# --- Streamlit UI ---
st.set_page_config(page_title="ESG Company Profile", layout="wide")
st.title("🌱 ESG Company Profile Generator")

company = st.text_input("Enter Company Name", value="TCS")
question = st.text_area("Enter ESG-related question", value="Generate ESG company profile")

if st.button("Generate Profile"):
    with st.spinner("Generating profile..."):
        result = qa_chain.invoke({"query": f"{company} {question}"})

        st.markdown("### ✅ ESG Profile")
        st.markdown(result['result'])

        st.markdown("### 📚 Sources")
        for i, doc in enumerate(result["source_documents"], 1):
            st.markdown(f"""
            **Source {i}**
            - 📄 **File**: {doc.metadata.get("source")}
            - 📅 **Year**: {doc.metadata.get("year")}
            - 📄 **Page**: {doc.metadata.get("page")}
            - 📝 **Snippet**: `{doc.page_content[:300]}`...
            """)

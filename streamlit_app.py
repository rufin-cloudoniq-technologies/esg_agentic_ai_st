# streamlit_app.py

import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# --- Streamlit Page Config ---
st.set_page_config(page_title="ESG Company Profile", layout="wide")
st.title("🌱 ESG Company Profile Generator")

# --- Configuration ---
VECTOR_STORE_PATH = os.path.join(".", "vector_store")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Cacheable Resource Loaders ---
@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

@st.cache_resource
def load_vectorstore(_embedding):
    return FAISS.load_local(VECTOR_STORE_PATH, embedding, allow_dangerous_deserialization=True)

@st.cache_resource
def load_llm():
    # 🧠 Use a local lightweight HuggingFace model (no token needed)
    local_pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_length=512
    )
    return HuggingFacePipeline(pipeline=local_pipe)

# --- Load Components ---
embedding = load_embedding()
vectorstore = load_vectorstore(embedding)
llm = load_llm()

# --- Prompt Template ---
prompt_template = PromptTemplate.from_template("""
You are a Financial Analyst tasked with providing clear, concise, and accurate answers using only the context provided.

Instructions:
- Analyze the content strictly within the financial context.
- Do NOT fabricate numbers or financial metrics.
- Be very specific. If only data for one year or one dimension is requested, then only search for that. If possible convert that into tabular format. If it is not found then mention that data is not found.
- Focus on company performance, ESG metrics, risks, opportunities, and trends if mentioned.
- Your answer must be data-grounded and business-relevant.

Context:
{context}

Question: {question}

Financial Analyst’s Answer:
""")

# --- Build QA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# --- Streamlit UI Form ---
company = st.text_input("🔍 Enter Company Name", value="TCS")
question = st.text_area("✏️ ESG-related Question", value="Generate ESG company profile")

if st.button("Generate Profile"):
    with st.spinner("🔄 Generating profile..."):
        query = f"{company} {question}"
        result = qa_chain.invoke({"query": query})

        st.subheader("✅ ESG Profile")
        st.markdown(result['result'])

        st.subheader("📚 Sources")
        for i, doc in enumerate(result["source_documents"], 1):
            st.markdown(f"""
**Source {i}**
- 📄 **File**: {doc.metadata.get("source")}
- 📅 **Year**: {doc.metadata.get("year")}
- 📄 **Page**: {doc.metadata.get("page")}
- 📝 **Snippet**: `{doc.page_content[:300]}`...
""")

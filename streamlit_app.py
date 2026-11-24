import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="Yamaha Keyboard Assistant", page_icon="🎹")
st.title("🎹 Yamaha Keyboard Manual Assistant")
st.caption("Ask questions in English, Telugu, Tenglish, or any regional language — powered by Vanilla RAG")

# Load Vector DB & LLM once (cached for speed)
@st.cache_resource
def load_rag_components():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # If chroma_db doesn't exist on the cloud server, build it from the PDF
    if not os.path.exists("./chroma_db"):
        loader = PyPDFLoader("data/psre473_en_om_a1 manual.pdf")
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    else:
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
    return db, llm

vector_db, llm = load_rag_components()

prompt_template = ChatPromptTemplate.from_template("""
You are a friendly assistant for a Yamaha Keyboard Manual.
Answer the user's question ONLY using the provided context below.

Language Rules:
- Always reply in the same language the user asks in (English, Telugu, Tenglish, Hindi, Tamil, etc.).
- If answering in Telugu, use simple, everyday spoken Telugu (వాడుక భాష) that anyone can easily understand. Do NOT use heavy, formal, or bookish Telugu words. Keep technical words like keyboard, volume, button, headphones, cable, and power in English (e.g., "వాల్యూమ్", "హెడ్‌ఫోన్స్").
- If the user asks in Tenglish (Telugu written in English), answer clearly in friendly, simple Tenglish.
- If the answer is not in the context, politely say you cannot find that information in the manual (in the same simple language).

Context:
{context}

Question:
{question}
""")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input box
if user_query := st.chat_input("Ask about your Yamaha keyboard (English / తెలుగు / Tenglish)..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 1. Translate query to English for accurate ChromaDB search
    translation_prompt = f"Translate the following question to simple English for search. If it is already in English, return it as is. Return ONLY the English text:\n{user_query}"
    english_query = llm.invoke(translation_prompt).content.strip()

    # 2. Retrieve relevant English chunks from ChromaDB
    retrieved_docs = vector_db.similarity_search(english_query, k=6)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 3. Generate answer in user's original language (Simple Spoken Telugu / Tenglish / English / Regional)
    final_prompt = prompt_template.format(context=context_text, question=user_query)

    with st.chat_message("assistant"):
        response = llm.invoke(final_prompt)
        st.markdown(response.content)

    st.session_state.messages.append({"role": "assistant", "content": response.content})
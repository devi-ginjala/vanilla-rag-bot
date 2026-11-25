import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="Yamaha Keyboard Assistant", page_icon="🎹")
st.title("🎹 Yamaha Keyboard Assistant")
st.caption("Ask anything about your PSR-E473 or ask to explain any page (1–76) in English, Telugu, or Tenglish.")

@st.cache_resource
def load_rag_components():
    loader = PyPDFLoader("data/psre473_en_om_a1 manual.pdf")
    pages = loader.load()

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    if not os.path.exists("./chroma_db"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    else:
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
    return db, llm, pages

vector_db, llm, pdf_pages = load_rag_components()

hybrid_prompt = ChatPromptTemplate.from_template("""
You are a friendly assistant for a Yamaha Keyboard Manual and general keyboard learning.

Answering Rules (Hybrid RAG):
1. First, check the provided Manual Context below. If the answer is in the context, answer accurately based on it.
2. If the answer is NOT in the manual, BUT the question is related to keyboards, piano, music, chords, scales, or practice tips:
   - Start your response with a clear note (in the user's language) saying: "This information is not in the manual, but based on general music knowledge:"
   - Then provide a helpful, beginner-friendly answer.
3. If the question is completely unrelated to keyboards or music, politely say you can only help with the Yamaha keyboard and music.

Language Rules:
- Always reply in the same language the user asks in (English, Telugu, Tenglish, Hindi, Tamil, etc.).
- If answering in Telugu, use simple, everyday spoken Telugu (వాడుక భాష). Do NOT use heavy, formal, or bookish Telugu words. Keep technical words like keyboard, volume, button, headphones, cable, chords, and power in English.
- If the user asks in Tenglish (Telugu written in English), answer clearly in friendly, simple Tenglish.

Manual Context:
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

# Single clean chat input (ChatGPT / Gemini style)
if user_query := st.chat_input("Ask a question or type 'Explain page 10 in Telugu'..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Smart Check: Did the user ask for a specific page number?
    page_match = re.search(r'(?:page|pg|పేజీ)\s*(?:no\.?|number|నంబర్)?\s*(\d+)', user_query, re.IGNORECASE)

    with st.chat_message("assistant"):
        if page_match:
            page_num = int(page_match.group(1))
            if 1 <= page_num <= len(pdf_pages):
                page_text = pdf_pages[page_num - 1].page_content.strip()
                if not page_text:
                    answer = f"Page {page_num} looks blank or only contains diagrams."
                else:
                    page_prompt = f"""
                    You are a friendly Yamaha Keyboard assistant.
                    The user asked: "{user_query}"
                    Explain the following manual page clearly in the SAME language the user requested (if Telugu, use simple everyday spoken Telugu and keep technical words like keyboard, button, volume, power in English).
                    Organize with clean bullet points.

                    Manual Page {page_num} Content:
                    {page_text}
                    """
                    answer = llm.invoke(page_prompt).content
            else:
                answer = f"This manual only has {len(pdf_pages)} pages. Please ask for a page between 1 and {len(pdf_pages)}."
        else:
            # Normal Hybrid RAG flow
            translation_prompt = f"Translate the following question to simple English for search. If it is already in English, return it as is. Return ONLY the English text:\n{user_query}"
            english_query = llm.invoke(translation_prompt).content.strip()

            retrieved_docs = vector_db.similarity_search(english_query, k=6)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

            final_prompt = hybrid_prompt.format(context=context_text, question=user_query)
            answer = llm.invoke(final_prompt).content

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
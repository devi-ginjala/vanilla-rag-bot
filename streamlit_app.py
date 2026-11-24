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
st.caption("Chat in Telugu/Tenglish/English OR learn the manual Page-by-Page")

# Load PDF pages, Vector DB & LLM once (cached for speed)
@st.cache_resource
def load_rag_components():
    # Always load pages so Page-by-Page Translator can read any exact page
    loader = PyPDFLoader("data/psre473_en_om_a1 manual.pdf")
    pages = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists("./chroma_db"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    else:
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
    return db, llm, pages

vector_db, llm, pdf_pages = load_rag_components()

# Mode switcher at the top (works smoothly on mobile & desktop)
mode = st.radio(
    "మీకు ఏది కావాలి?",
    ["💬 Chat Assistant", "📖 Page-by-Page Translator"],
    horizontal=True
)

st.divider()

# ==========================================
# MODE 1: HYBRID RAG CHAT ASSISTANT
# ==========================================
if mode == "💬 Chat Assistant":
    prompt_template = ChatPromptTemplate.from_template("""
    You are a friendly assistant for a Yamaha Keyboard Manual and general keyboard learning.

    Answering Rules (Hybrid RAG):
    1. First, check the provided Manual Context below. If the answer is in the context, answer accurately based on it.
    2. If the answer is NOT in the manual, BUT the question is related to keyboards, piano, music, chords, scales, or practice tips:
       - Start your response with a clear note (in the user's language) saying: "This information is not in the manual, but based on general music knowledge:"
       - Then provide a helpful, beginner-friendly answer.
    3. If the question is completely unrelated to keyboards or music (e.g., cooking, politics, coding), politely say you can only help with the Yamaha keyboard and music.

    Language Rules:
    - Always reply in the same language the user asks in (English, Telugu, Tenglish, Hindi, Tamil, etc.).
    - If answering in Telugu, use simple, everyday spoken Telugu (వాడుక భాష). Do NOT use heavy, formal, or bookish Telugu words. Keep technical words like keyboard, volume, button, headphones, cable, chords, and power in English.
    - If the user asks in Tenglish (Telugu written in English), answer clearly in friendly, simple Tenglish.

    Manual Context:
    {context}

    Question:
    {question}
    """)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Ask about your Yamaha keyboard or music tips (English / తెలుగు / Tenglish)..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 1. Translate query to English for accurate ChromaDB search
        translation_prompt = f"Translate the following question to simple English for search. If it is already in English, return it as is. Return ONLY the English text:\n{user_query}"
        english_query = llm.invoke(translation_prompt).content.strip()

        # 2. Retrieve top 6 chunks
        retrieved_docs = vector_db.similarity_search(english_query, k=6)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 3. Generate answer
        final_prompt = prompt_template.format(context=context_text, question=user_query)

        with st.chat_message("assistant"):
            response = llm.invoke(final_prompt)
            st.markdown(response.content)

        st.session_state.messages.append({"role": "assistant", "content": response.content})

# ==========================================
# MODE 2: PAGE-BY-PAGE TRANSLATOR
# ==========================================
elif mode == "📖 Page-by-Page Translator":
    st.subheader("📖 మాన్యువల్‌ని పేజీ-బై-పేజీ నేర్చుకోండి")
    st.write(f"మొత్తం పేజీలు: **{len(pdf_pages)}**. మీకు అర్థం కావాలనుకున్న పేజీ నంబర్ ఎంచుకోండి:")

    col1, col2 = st.columns(2)
    with col1:
        page_num = st.number_input(
            "Page Number:",
            min_value=1,
            max_value=len(pdf_pages),
            value=1,
            step=1
        )
    with col2:
        target_lang = st.selectbox(
            "ఏ భాషలో కావాలి?",
            ["Simple Telugu (వాడుక తెలుగు)", "Tenglish", "Simple English"]
        )

    if st.button("✨ ఈ పేజీని ఎక్స్‌ప్లెయిన్ చెయ్", type="primary"):
        page_text = pdf_pages[page_num - 1].page_content.strip()

        if not page_text:
            st.warning("ఈ పేజీలో టెక్స్ట్ లేదు (బహుశా కేవలం బొమ్మలు లేదా ఖాళీ పేజీ అయి ఉండవచ్చు).")
        else:
            with st.spinner(f"Page {page_num} ని {target_lang} లోకి మారుస్తున్నాను..."):
                page_prompt = f"""
                You are a friendly Yamaha Keyboard teacher.
                Translate and explain the following page from the Yamaha Keyboard Manual in {target_lang}.

                Rules:
                - Use very simple, beginner-friendly, everyday spoken language.
                - If using Telugu, use natural spoken Telugu (వాడుక భాష), NOT bookish Telugu. Keep technical terms like keyboard, button, power, volume, jack, display, chords in English.
                - Organize the explanation cleanly with bullet points so it is easy to read alongside the physical book.

                Manual Page {page_num} Content:
                {page_text}
                """
                response = llm.invoke(page_prompt)
                st.markdown(f"### 📄 Page {page_num} Explanation")
                st.markdown(response.content)

                with st.expander("🔍 ఒరిజినల్ ఇంగ్లీష్ టెక్స్ట్ చూడండి"):
                    st.text(page_text)
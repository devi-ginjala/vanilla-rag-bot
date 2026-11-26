# Vanilla RAG Document Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built from scratch to answer technical queries from a 76-page Yamaha Keyboard user manual with strict hallucination control.

## Live Demo
- Streamlit App: https://vanilla-rag-bot.streamlit.app/

## Tech Stack
- Language: Python
- Framework: LangChain
- Document Parsing: PyPDF (with AES decryption support via cryptography)
- Embeddings: HuggingFace (all-MiniLM-L6-v2 — local, zero-cost)
- Vector Database: ChromaDB (persistent local storage)
- LLM: Groq API (openai/gpt-oss-20b, temperature=0)

## Architecture
1. Ingestion & Chunking (app.py): Loads the 76-page PDF and splits it into 253 overlapping chunks (chunk_size=1000, chunk_overlap=200) using RecursiveCharacterTextSplitter.
2. Embedding & Storage (app.py): Converts chunks into dense vectors using all-MiniLM-L6-v2 and persists them in chroma_db/.
3. Retrieval & Q&A (chat.py): Loads the persisted ChromaDB, retrieves the top-3 relevant chunks via similarity search, and passes them to the LLM with a strict grounding prompt to reject out-of-context queries.

## How to Run
1. Install dependencies: pip install -r requirements.txt
2. Add your Groq API key to a .env file: GROQ_API_KEY=your_key_here
3. Build the vector database: python app.py
4. Start chatting: python chat.py

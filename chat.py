import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Load API key from .env file
load_dotenv()

# 2. Load the saved Vector Database (No need to re-process the PDF!)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# 3. Initialize the LLM (Groq - Llama 3)
llm = ChatGroq(
    model_name="openai/gpt-oss-20b",
    temperature=0
)

# 4. Create a strict System Prompt (Prevents Hallucination)
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant for a Yamaha Keyboard Manual.
Answer the user's question ONLY using the provided context below.
If the answer is not in the context, say "I cannot find that information in the manual."

Context:
{context}

Question:
{question}
""")

# 5. Interactive Chat Loop
print("🤖 Vanilla RAG Bot is ready! Type 'exit' to quit.\n")

while True:
    user_query = input("You: ")
    if user_query.lower() == "exit":
        break

    # Retrieve top 3 relevant chunks
    retrieved_docs = vector_db.similarity_search(user_query, k=3)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Format prompt and get response from LLM
    final_prompt = prompt_template.format(context=context_text, question=user_query)
    response = llm.invoke(final_prompt)

    print(f"\nBot: {response.content}\n")
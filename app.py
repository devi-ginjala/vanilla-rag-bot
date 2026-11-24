from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma 

# 1. Load PDF document (Data Ingestion)
loader = PyPDFLoader("data/psre473_en_om_a1 manual.pdf")
docs = loader.load()
print(f"Total Pages Loaded: {len(docs)}")

# 2. Split document into smaller chunks (Chunking)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(docs)
print(f"Total Chunks Created: {len(chunks)}")

# Preview the first chunk
print("\n--- Sample Chunk 1 ---")
print(chunks[0].page_content)

# 3. Initialize Embedding Model (Text to Vectors)
print("\nLoading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Store Chunks in ChromaDB (Vector Database)
print("Creating vector database...")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)
print("Vector database created and saved successfully!")

# 5. Test Retrieval (Similarity Search)
query = "What are the precautions before using the instrument?"
retrieved_docs = vector_db.similarity_search(query, k=2)

print(f"\n--- Top 2 Relevant Chunks for: '{query}' ---")
for i, doc in enumerate(retrieved_docs):
    print(f"\nResult {i+1}:")
    print(doc.page_content)
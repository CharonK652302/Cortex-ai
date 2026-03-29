from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks):
    print("🔄 Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    print("📦 Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("vector_db")

    print("✅ Vector DB saved successfully!")
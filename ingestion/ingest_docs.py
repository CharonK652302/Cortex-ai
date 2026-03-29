import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.vector_store import create_vector_store


def load_documents(folder_path):
    documents = []

    print("Checking folder:", folder_path)

    for file in os.listdir(folder_path):
        print("Found file:", file)

        if file.endswith(".pdf"):
            print("Loading PDF:", file)

            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs = loader.load()
            documents.extend(docs)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    docs = load_documents("data\documents")

    print(f"Loaded {len(docs)} pages")

    chunks = split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    create_vector_store(chunks)
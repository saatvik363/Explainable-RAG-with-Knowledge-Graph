import os
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

def retriever():
    # Embedding setup 
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    # Check if database exists
    db_exists = os.path.exists("faiss_index")

    if db_exists:
        update = input("Database found! Do you want to update it? (y/n): ")
    else:
        update = "y"

    if update.lower() == "y":
        filepath = input("What is the file path: ")

        # Document loaders
        loader = GenericLoader(
            blob_loader=FileSystemBlobLoader(
                path=filepath,
                glob="*.pdf",
            ),
            blob_parser=PyPDFParser(),
        )
        document = loader.load()

        # Split in chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

        final_chunks = text_splitter.split_documents(document)

        # Create and Save
        print("Creating vector database...")
        db = FAISS.from_documents(final_chunks, hf)
        db.save_local("faiss_index")
        print("Database saved.")
        
    else:
        print("Loading existing database...")
        # Load from disk
        db = FAISS.load_local("faiss_index", hf, allow_dangerous_deserialization=True)

    # Search
    query = input("What is question you want to ask: ")
    results = db.similarity_search(query, 3)

    return query, results

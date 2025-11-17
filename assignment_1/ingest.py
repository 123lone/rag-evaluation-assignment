import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

CORPUS_DIR = "corpus"
PERSIST_DIR = "chroma_store"

def ingest():
    os.makedirs(PERSIST_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection("docs")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    files = glob.glob(os.path.join(CORPUS_DIR, "*.txt"))
    for file_path in files:
        text = open(file_path, "r", encoding="utf-8").read()
        chunks = splitter.split_text(text)
        embeddings = embedder.encode(chunks)

        base = os.path.basename(file_path)
        for i, chunk in enumerate(chunks):
            collection.add(
                ids=[f"{base}-{i}"],
                documents=[chunk],
                embeddings=[embeddings[i].tolist()]  # required for Chroma
            )

    print("Ingestion complete.")

if __name__ == "__main__":
    ingest()

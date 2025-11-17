import chromadb
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, persist_dir="chroma_store"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection("docs")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query, k=5):
        q_emb = self.embedder.encode([query])[0].tolist()   # convert to list
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k
        )
        return results

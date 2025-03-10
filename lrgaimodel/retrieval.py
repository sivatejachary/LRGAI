import faiss
import numpy as np

class Retriever:
    def __init__(self, embedding_dim=256):
        self.index = faiss.IndexFlatL2(embedding_dim)
    
    def add_documents(self, embeddings):
        self.index.add(np.array(embeddings, dtype=np.float32))
    
    def retrieve(self, query_embedding, k=5):
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
        return indices[0]

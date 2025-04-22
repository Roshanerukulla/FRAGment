# flashrag/retriever/faiss_retriever.py (CPU-only)

import faiss
import pickle
from sentence_transformers import SentenceTransformer

class FaissRetriever:
    def __init__(
        self,
        index_path="retriever/faiss_index.index",
        doc_path="retriever/doc_texts.pkl",
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        top_k=5,
        similarity_threshold=0.3,
    ):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # Load FAISS index (CPU-only)
        self.index = faiss.read_index(index_path)

        # Load document chunks
        with open(doc_path, "rb") as f:
            self.documents = pickle.load(f)

        # Embedder (defaults to GPU if available)
        self.embedder = SentenceTransformer(embed_model_name)

    def retrieve(self, query):
        query_vec = self.embedder.encode([query])
        distances, indices = self.index.search(query_vec, self.top_k)

        retrieved_docs = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            if idx < len(self.documents):
                doc = self.documents[idx]
                similarity = score
                if similarity >= self.similarity_threshold:
                    doc["similarity"] = similarity
                    retrieved_docs.append(doc)

        return retrieved_docs

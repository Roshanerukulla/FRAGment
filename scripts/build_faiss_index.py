import os
import json
import faiss
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import ijson
import numpy as np

def main():
    chunk_path = "retriever/doc_texts.json"
    index_path = "retriever/faiss_index.index"
    pickle_path = "retriever/doc_texts.pkl"

    print(" Streaming chunks from:", chunk_path)

    # Use streaming JSON parser to avoid memory overload
    context_chunks = []
    with open(chunk_path, "r", encoding="utf-8") as f:
        for chunk in ijson.items(f, "item"):
            context_chunks.append(chunk)

    print(f" Loaded {len(context_chunks):,} context chunks.")

    # Load embedder
    print(" Loading embedding model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print(" Embedding in batches (GPU-accelerated if available)...")
    batch_size = 1024
    embeddings = []
    texts = [doc["text"] for doc in context_chunks]

    for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
        batch_texts = texts[i:i + batch_size]
        batch_embeds = embedder.encode(batch_texts, show_progress_bar=False)
        embeddings.append(batch_embeds)

    embeddings = np.vstack(embeddings)
    print(f" Total Embeddings Shape: {embeddings.shape}")

    # FAISS Index creation
    dim = embeddings.shape[1]
    print("📦 Creating FAISS FlatL2 index (accurate, large)...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save
    faiss.write_index(index, index_path)
    with open(pickle_path, "wb") as f:
        pickle.dump(context_chunks, f)

    print(" FAISS index built and saved successfully.")

if __name__ == "__main__":
    main()

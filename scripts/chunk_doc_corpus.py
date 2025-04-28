

import json
from tqdm import tqdm
import argparse
import os

from sentence_transformers import SentenceTransformer


def chunk_text(text, chunk_size=3):
   
    sentences = [s.strip() for s in text.replace("\n", " ").split('.') if s.strip()]

    
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        if len(chunk.strip()) > 10:
            chunks.append(chunk.strip())
    return chunks

def process_hotpotqa(input_path, output_path, limit=None):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_chunks = []

    for i, item in enumerate(tqdm(data[:limit] if limit else data)):
        for title, paragraphs in item.get("context", []):
            for paragraph in paragraphs:
                for chunk in chunk_text(paragraph):
                    all_chunks.append({"title": title, "text": chunk})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(all_chunks)} chunks to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="hotpot_data/hotpot_train_v1.1.json", help="Path to original HotpotQA")
    parser.add_argument("--output", default="retriever/doc_texts.json", help="Path to save chunked docs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples (for debugging)")
    args = parser.parse_args()

    process_hotpotqa(args.input, args.output, args.limit)

# 🧠 FRAGment: A Refined FlashRAG Pipeline for Multi-hop Question Answering

FRAGment is an enhanced version of the FlashRAG pipeline for multi-hop QA. It improves document retrieval, reranking, and generation using semantic techniques and modular upgrades. Built with FAISS, Cohere, and a subset of HotpotQA.

---

##  Features

- **FAISS-based dense retrieval** using Sentence-BERT
- **Cohere Reranker** to sort documents by semantic relevance
- **Cohere Generator** (command-r-plus) for high-quality answer generation
- **Improved chunking** to remove noise and short irrelevant segments
- **Evaluation support** with EM, F1, Precision, and Recall metrics
- **Gradio UI** with conversational memory for debugging and testing

---

# Setup Instructions

### 1. Clone the repository

git clone https://github.com/YOUR_USERNAME/FRAGment.git
cd FRAGment


### 2. Create & activate virtual environment

python -m venv frag_env
source frag_env/bin/activate  # Linux/macOS
frag_env\Scripts\activate    # Windows


### 3. Install dependencies

pip install -r requirements.txt


---

## 🧾 Data: HotpotQA v1.1

FRAGment uses a subset of the [HotpotQA](https://hotpotqa.github.io/) dataset for training and evaluation.

###  Steps to Download & Prepare:

1. **Download the official HotpotQA dataset:**

wget https://rajpurkar.github.io/files/hotpotqa/hotpot_train_v1.1.json


2. **Create data directory and move file:**

mkdir hotpot_data
mv hotpot_train_v1.1.json hotpot_data/


3. **Chunk the dataset (use limit for faster testing):**

python scripts/chunk_doc_corpus.py --limit 15000


4. **Build FAISS index:**

python scripts/build_faiss_index.py


---

##  Evaluation

Run evaluation script after generating answers:

python scripts/eval_results.py



---



---

## 🎮 Run the Web App

python -m scripts.web_app


---

##  Acknowledgements

- FlashRAG base: https://github.com/thunlp/FlashRAG
- HotpotQA dataset: https://hotpotqa.github.io
- FAISS: https://github.com/facebookresearch/faiss
- Cohere: https://cohere.com

---


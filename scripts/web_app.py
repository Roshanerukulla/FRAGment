import os
import gradio as gr
from dotenv import load_dotenv
from collections import deque

from flashrag.generator.cohere_generator import CohereGenerator
from flashrag.retriever.faiss_retriever import FaissRetriever
from flashrag.reranker.cohere_reranker import CohereReranker

load_dotenv()

# Initialize pipeline components
retriever = FaissRetriever(similarity_threshold=0.3)
reranker = CohereReranker(api_key=os.getenv("COHERE_API_KEY"))
generator = CohereGenerator(api_key=os.getenv("COHERE_API_KEY"))

# Store last 10 turns of conversation
conversation_history = deque(maxlen=10)

def answer_question(query):
    # Retrieve documents
    retrieved_chunks = retriever.retrieve(query)

    if not retrieved_chunks:
        return " No relevant documents found.", " No relevant documents found.", "(no conversation yet)"

    # Rerank retrieved documents
    reranked_docs_with_scores = reranker.rerank(query, retrieved_chunks)
    top_reranked_docs = [doc for doc, _ in reranked_docs_with_scores]

    # Generate answer
    answer = generator.generate(query, top_reranked_docs)

    # Store conversation
    conversation_history.append((query, answer))

    # Prepare debug chunk preview
    debug_chunks = "\n\n".join([
        f"({score:.2f}) {doc['text'][:400]}..." 
        for doc, score in reranked_docs_with_scores[:5] 
        if doc is not None and isinstance(doc, dict) and 'text' in doc
    ])

    # Format conversation history
    history_text = "\n\n".join([
        f" {q}\n {a}" for q, a in conversation_history
    ])

    return answer, debug_chunks, history_text

# Launch gradio UI
ui = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask your question here...", label="💬 Query", container=True),
    outputs=[
        gr.Textbox(label="🤖 Answer", lines=3),
        gr.Textbox(label="📄 Top Retrieved Chunks (reranked with score)", lines=10),
        gr.Textbox(label="🕰️ Conversation History", lines=10)
    ],
    title="🔍 Ask Me Anything (HotpotQA + FlashRAG + Reranker)",
    description="Multi-step RAG pipeline using FAISS + Cohere Embed + Cohere Rerank + Cohere Generate. Now with conversation memory!",
    theme="default"
)

if __name__ == "__main__":
    ui.launch()

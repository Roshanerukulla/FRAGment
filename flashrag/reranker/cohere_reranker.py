import os
import cohere
from typing import List, Tuple, Union
from dotenv import load_dotenv

load_dotenv()

class CohereReranker:
    def __init__(self, api_key: str = None, top_k: int = 5):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.client = cohere.Client(self.api_key)
        self.top_k = top_k

    def rerank(self, query: str, docs: List[Union[str, dict]]) -> List[Tuple[dict, float]]:
        if not docs:
            return []

        # Ensure we're passing plain text documents to Cohere
        doc_texts = [doc["text"] if isinstance(doc, dict) else doc for doc in docs]

        response = self.client.rerank(
            model="rerank-english-v2.0",
            query=query,
            documents=doc_texts,
            top_n=min(self.top_k, len(doc_texts))
        )

        reranked = [
            (docs[result.index], result.relevance_score) for result in response.results
            if docs[result.index] is not None and isinstance(docs[result.index], dict)
        ]
        return reranked

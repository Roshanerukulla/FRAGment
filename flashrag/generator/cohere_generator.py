import cohere
from typing import List, Union

class CohereGenerator:
    def __init__(self, api_key: str, model: str = "command-r-plus", max_tokens: int = 300):
        self.client = cohere.Client(api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, question: str, context_docs: List[Union[str, dict]]) -> str:
        context = "\n".join([
            doc["text"] if isinstance(doc, dict) and doc.get("text") else str(doc)
            for doc in context_docs
            if isinstance(doc, str) or (isinstance(doc, dict) and doc.get("text"))
        ])

        prompt = f"""Answer the following multi-hop question based on the context:

Context:
{context}

Question:
{question}

Answer:"""

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=0.7
        )

        return response.generations[0].text.strip()

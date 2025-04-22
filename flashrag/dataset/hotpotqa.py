import json
from torch.utils.data import Dataset

class HotpotQADataset(Dataset):
    def __init__(self, data_path: str, max_samples: int = None, chunk_by: str = "sentence"):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        if max_samples:
            self.data = self.data[:max_samples]

        self.chunk_by = chunk_by

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        context = item["context"]

        # Split the context string into "docs" based on your chunking strategy
        if self.chunk_by == "sentence":
            docs = [sent.strip() for sent in context.split(". ") if sent.strip()]
        elif self.chunk_by == "paragraph":
            docs = [para.strip() for para in context.split("\n\n") if para.strip()]
        else:
            docs = [context.strip()]  # fallback

        return {
            "query": question,
            "docs": docs,
            "answer": item.get("answer", "")
        }

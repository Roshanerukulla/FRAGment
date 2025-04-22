import json
import difflib

# Load raw HotpotQA with answers
with open("hotpot_data/hotpot_train_v1.1.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Build a map of question → answer
answer_map = {item["question"]: item["answer"] for item in raw_data}
all_questions = list(answer_map.keys())

# Load preprocessed
with open("hotpot_data/preprocessed_hotpot.json", "r", encoding="utf-8") as f:
    preprocessed = json.load(f)

missed = 0
for item in preprocessed:
    q = item["question"]
    # Try direct match
    if q in answer_map:
        item["answer"] = answer_map[q]
    else:
        # Try fuzzy match
        match = difflib.get_close_matches(q, all_questions, n=1, cutoff=0.9)
        if match:
            item["answer"] = answer_map[match[0]]
        else:
            item["answer"] = ""
            missed += 1

# Save updated file
with open("hotpot_data/preprocessed_hotpot_with_answers.json", "w", encoding="utf-8") as f:
    json.dump(preprocessed, f, indent=2, ensure_ascii=False)

print(f" Done: Added answers with fuzzy matching. Missed {missed} questions.")

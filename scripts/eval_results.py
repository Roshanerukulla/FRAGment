import os
import sys
import json
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flashrag.dataset import Dataset
from flashrag.evaluator.evaluator import Evaluator

def main():
    load_dotenv()

    # Load config
    with open("eval_config.json", "r") as f:
        config = json.load(f)

    # ✅ Make sure output folder exists
    os.makedirs(config["save_dir"], exist_ok=True)

    # ✅ Load dataset with predictions
    dataset = Dataset(
        dataset_path="hotpot_data/generated_predictions.json",
        sample_num=None
    )

    # 🧪 Debug: See predictions vs ground truth
    for i, item in enumerate(dataset[:5]):
        print(f"\nQ: {item.question}")
        print(f"🔮 Pred: {item.output.get('pred', '')}")
        print(f"🎯 Gold: {item.golden_answers}")

    # ✅ Evaluate
    evaluator = Evaluator(config)
    result = evaluator.evaluate(dataset)

    print("\n📊 Evaluation Results:")
    for metric, score in result.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()

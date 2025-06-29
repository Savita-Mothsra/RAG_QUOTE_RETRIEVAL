from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset  # ✅ new import
from dotenv import load_dotenv
import os

# 🔐 Load OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ✅ Your examples as list of dicts
examples = [
    {
        "question": "Quotes about insanity attributed to Einstein",
        "answer": "Insanity is doing the same thing over and over again and expecting different results.",
        "contexts": [
            "Insanity is doing the same thing over and over again and expecting different results. - Albert Einstein"
        ],
        "ground_truth": "Insanity is doing the same thing over and over again and expecting different results."
    },
    {
        "question": "Motivational quotes tagged ‘accomplishment’",
        "answer": "Accomplishment is the start of a greater journey.",
        "contexts": [
            "Accomplishment is the start of a greater journey. - Unknown"
        ],
        "ground_truth": "Accomplishment is the start of a greater journey."
    }
]

# ✅ Convert list to HuggingFace Dataset
dataset = Dataset.from_list(examples)

# ✅ Evaluate
scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)

print("\n--- RAG Evaluation Scores ---")
print(scores)






import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load Dataset
dataset = load_dataset("Abirate/english_quotes", split="train")

# Load a base sentence transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Create training examples: quote paired with a query (author + tags)
train_examples = []

for item in dataset:
    quote = item["quote"]
    author = item["author"]
    tags = ", ".join(item["tags"]) if item["tags"] else "quotes"

    query = f"{tags} quote by {author}"  # Simulated semantic query
    train_examples.append(InputExample(texts=[quote, query], label=1.0))

# DataLoader and Loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    show_progress_bar=True
)

# Save the fine-tuned model
model_save_path = "models/fine_tuned_model"
os.makedirs(model_save_path, exist_ok=True)
model.save(model_save_path)

print(f"✅ Model saved to {model_save_path}")



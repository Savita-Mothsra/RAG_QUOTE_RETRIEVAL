from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

model = SentenceTransformer("models/fine_tuned_model")
with open("data/cleaned_quotes.json", "r") as f:
    data = json.load(f)
corpus = [d['full_text'] for d in data]
embeddings = model.encode(corpus, show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
faiss.write_index(index, "models/faiss_index.idx")
with open("models/data_index.json", "w") as f:
    json.dump(data, f)
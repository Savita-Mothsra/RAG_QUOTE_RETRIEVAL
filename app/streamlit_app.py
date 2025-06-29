import streamlit as st
import json
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# 🌐 Get absolute paths based on current script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/fine_tuned_model")
INDEX_PATH = os.path.join(BASE_DIR, "../models/faiss_index.idx")
DATA_PATH = os.path.join(BASE_DIR, "../models/data_index.json")

# 🔠 Title
st.title("📖 Semantic Quote Search")

# 🔄 Load Model & Data
model = SentenceTransformer(MODEL_PATH)
index = faiss.read_index(INDEX_PATH)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# 🔍 Search UI
query = st.text_input("Enter your query (e.g., Quotes about courage by women authors):")

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        query_embedding = model.encode([query])
        D, I = index.search(np.array(query_embedding), 5)
        results = [data[i] for i in I[0]]

        st.json({"query": query, "results": results})
        st.write("**Similarity Scores:**", D[0])
        st.download_button("Download JSON", json.dumps(results, indent=2), file_name="results.json")

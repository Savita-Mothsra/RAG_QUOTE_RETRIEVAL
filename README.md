# Video Link : https://drive.google.com/file/d/1AgLsSf8iKcwFDe2tDFPx5mm_ncPs2O1Q/view?usp=sharing
#  Semantic Quote Retrieval with RAG

This project implements a **Retrieval Augmented Generation (RAG)** system to semantically search quotes from the [Abirate/english_quotes]dataset using a fine-tuned sentence embedding model and FAISS, with a Streamlit interface and evaluation using RAGAS.

---

##  Features

-  **Semantic Search**: Retrieve quotes based on natural language queries.
-  **Fine-tuned Model**: SentenceTransformer adapted on quotes + author + tags.
-  **FAISS Indexing**: Efficient retrieval using vector similarity.
-  **RAG Evaluation**: Automated metrics using RAGAS.
-  **Streamlit UI**: Interactive app with query input, JSON response, and downloads.

---

##  Project Structure

```
.
├── 01_data_preparation.py     # Clean and preprocess dataset
├── 02_model_finetuning.py     # Fine-tune SentenceTransformer on quote-author-tag pairs
├── 03_rag_pipeline.py         # Embed quotes and build FAISS index
├── 04_rag_evaluation.py       # Evaluate using RAGAS metrics
├── streamlit_app.py           # Streamlit interface for querying
├── models/                    # Saved fine-tuned model and FAISS index
└── data/
    └── cleaned_quotes.json    # Processed quote dataset
```

---

##  How to Run

1. **Install Requirements**:
   
   pip install -r requirements.txt
   

2. **Step-by-Step Execution**:
   - Prepare data:  
     
     python 01_data_preparation.py
     
   - Fine-tune model:  
     
     python 02_model_finetuning.py
     
   - Build RAG pipeline:  
     
     python 03_rag_pipeline.py
     
   - Evaluate RAG:  
    
     python 04_rag_evaluation.py
     

3. **Run Streamlit App**:
   
   streamlit run streamlit_app.py
   

---

##  Example Queries

- “Quotes about insanity attributed to Einstein”
- “Motivational quotes tagged ‘accomplishment’”
- “All Oscar Wilde quotes with humor”

---

##  Evaluation

Evaluated using **RAGAS** with the following metrics:

- Faithfulness
- Answer Relevancy
- Context Precision

Evaluation results are printed in `04_rag_evaluation.py`.

---

##  Requirements

```
transformers
datasets
sentence-transformers
faiss-cpu
numpy
streamlit
ragas
python-dotenv
```


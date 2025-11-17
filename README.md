# RAG Evaluation Assignment – Gemini Version

This repository contains:
- Assignment 1 → RAG System (Chunking, Embeddings, Retrieval)
- Assignment 2 → Evaluation Metrics (Precision, Recall, F1, MRR, Hit Rate)
- Streamlit UI → Query the system with Gemini-powered RAG

## How to Run

### 1. Create venv
py -3.11 -m venv venv
venv\Scripts\activate

### 2. Install dependencies
pip install -r requirements.txt

### 3. Add your Gemini API Key
Create file `.env`:
GEMINI_API_KEY=YOUR_KEY

### 4. Ingest Corpus
python assignment_1/ingest.py

### 5. Run App
streamlit run streamlit_app/app.py

### 6. Evaluate
python assignment_2/evaluation.py


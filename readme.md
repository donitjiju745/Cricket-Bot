# 🏏 Cricket QA & Dialogue Management System

A hybrid Question Answering and chatbot prototype combining:
- **Information Retrieval (IR):** TF-IDF passage retrieval + local extractive span extraction (MiniLM).
- **Knowledge Base (KB):** Structured table query resolution with Pandas.
- **Dialogue Management:** Intent routing and conversational frame tracking.

## Setup
1. Clone repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the MiniLM QA model into `models/minilm/` from Hugging Face (`deepset/minilm-uncased-squad2`).
4. Run dashboard: `streamlit run app.py`
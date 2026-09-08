# 🏏 Cricket QA & Dialogue Management System

A lightweight, local hybrid Question Answering (QA) and chatbot prototype for cricket-related questions. The system combines Information Retrieval (IR), extractive Question Answering, structured knowledge-base querying, and simple dialogue management in a single Streamlit application.

## 🎯 Project Objective

The system answers cricket-related questions through two complementary pipelines:

- **IR + Extractive QA:** Retrieves relevant cricket passages using TF-IDF and cosine similarity, then extracts a factoid answer using a locally loaded MiniLM-SQuAD2 model.
- **Structured KB QA:** Queries player and team statistics stored in CSV files using Pandas.
- **Dialogue Management:** Detects simple intents, tracks slots/context in a dialogue frame, and routes the query to the appropriate pipeline.

The complete system runs locally and does not require an OpenAI API or another external LLM API during normal operation.

## ✨ Features

- 💬 Interactive cricket chatbot
- 🔎 TF-IDF-based passage retrieval
- 🤖 Local Hugging Face extractive QA
- 📊 Structured player/team statistics lookup
- 🔄 Simple intent routing and dialogue-frame tracking
- 📋 Built-in sample QA benchmark
- 📈 Evaluation using Exact Match (EM), token-level F1, and latency
- 🧭 Multi-tab Streamlit dashboard for demonstrating each deliverable

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   User Question     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Preprocessing &      │
                    │ Intent Classification│
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
            Statistical Query       Text / Factoid Query
                    │                     │
                    ▼                     ▼
          ┌────────────────┐    ┌────────────────────┐
          │ Structured KB  │    │ TF-IDF Retrieval   │
          │   (Pandas/CSV) │    │ + Cosine Similarity│
          └───────┬────────┘    └──────────┬─────────┘
                  │                        │
                  │                        ▼
                  │              ┌────────────────────┐
                  │              │ Local MiniLM QA    │
                  │              │ Extractive Answer  │
                  │              └──────────┬─────────┘
                  │                        │
                  └───────────┬────────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Dialogue Frame /    │
                    │ Response Formatting  │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Dashboard │
                    └─────────────────────┘
```

## 🧰 Tools & Technologies

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.8+ | Core implementation |
| Frontend | Streamlit | Interactive web interface |
| IR | Scikit-learn | TF-IDF vectorization and cosine similarity |
| Extractive QA | Hugging Face Transformers | Local question-answering model |
| Deep Learning | PyTorch | Transformer model execution and tensor operations |
| Data Processing | Pandas | CSV loading, filtering, and evaluation |
| Storage | CSV files | Structured cricket data and benchmark data |
| Version Control | Git / GitHub | Source-code management |

## 📁 Project Structure

```text
Cricket-Bot/
│
├── app.py
├── logic.py
├── requirements.txt
├── readme.md
├── .gitignore
│
├── data/
│   ├── players_stats.csv
│   ├── teams_stats.csv
│   ├── sample_qa.csv
│   └── corpus/
│       ├── final_match.txt
│       ├── semifinal.txt
│       ├── tournament_rules.txt
│       └── venue_pitch.txt
│
└── models/
    └── minilm/
        └── [local MiniLM-SQuAD2 model files]
```

> Large model binaries should remain local and should not be committed to GitHub unless your course submission rules specifically require them. The repository uses `.gitignore` for large local model files.

## ⚙️ Requirements

Recommended environment:

- Python **3.8 or newer**
- pip
- Git
- Enough local disk space and RAM for the Transformer QA model

Python dependencies are listed in:

```text
requirements.txt
```

Core packages include:

```text
streamlit
pandas
scikit-learn
torch
transformers
```

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/donitjiju745/Cricket-Bot.git
cd Cricket-Bot
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the local QA model

The application expects the local extractive QA model in:

```text
models/minilm/
```

Model:

```text
deepset/minilm-uncased-squad2
```

Place the required tokenizer and model files in that directory before running the application.

### 5. Verify the data directory

Make sure the repository contains the structured CSV files and text corpus under:

```text
data/
```

The application uses these local files for structured statistics, retrieval, and benchmarking.

## ▶️ Running the Application

Launch the Streamlit frontend with:

```bash
streamlit run app.py
```

Streamlit will display a local URL, typically similar to:

```text
http://localhost:8501
```

Open that address in a browser.

### Backend vs Frontend

This project uses a **Streamlit-integrated Python architecture**, so there is no separate Flask/Node frontend server and no separate REST backend process.

- `logic.py` → core backend logic
- `app.py` → Streamlit frontend and application orchestration

Therefore, the complete application is started with:

```bash
streamlit run app.py
```

## 🖥️ Dashboard Tabs

### 1. 💬 Chatbot Prototype

The main chatbot interface allows users to:

- Enter cricket questions
- Receive generated answers
- View detected intent
- View selected pipeline
- Inspect the current dialogue frame and slots

### 2. 🔎 IR Passage & Factoid QA

This tab demonstrates:

- User query processing
- TF-IDF retrieval
- Top-ranked passages
- Extracted factoid answer
- QA confidence
- Response latency

### 3. 📊 Structured KB Query

This tab demonstrates deterministic lookup of:

- Player statistics
- Team statistics
- Retrieved database records
- Generated structured answers

### 4. 📋 Sample QA Dataset

Displays the benchmark questions and expected answers used for evaluation.

### 5. 🔄 Dialogue Flow & Frame Design

Displays the hybrid routing logic and the JSON-like structure used to track conversational state.

### 6. 📈 Evaluation Summary

Runs the evaluation benchmark and reports:

- Exact Match (EM)
- Token-level F1-score
- Average latency
- Prediction vs. ground-truth comparison

## 🧠 How the QA Pipeline Works

### IR Pipeline

1. Load text documents from the local corpus.
2. Convert documents to TF-IDF vectors.
3. Convert the user question to a TF-IDF vector.
4. Calculate cosine similarity.
5. Select the top relevant passages.
6. Pass the retrieved context and question to the local MiniLM-SQuAD2 model.
7. Extract the most probable answer span.

Conceptually:

```text
Question
   ↓
TF-IDF Vectorization
   ↓
Cosine Similarity
   ↓
Top-k Passages
   ↓
MiniLM Extractive QA
   ↓
Factoid Answer
```

### Structured KB Pipeline

1. Convert the query to lowercase.
2. Detect statistical keywords.
3. Identify the player or team entity.
4. Search the corresponding Pandas dataframe.
5. Extract the requested statistic.
6. Format the result as a natural-language answer.

Conceptually:

```text
Question
   ↓
Intent / Keyword Detection
   ↓
Player or Team Matching
   ↓
Pandas CSV Query
   ↓
Statistic Retrieval
   ↓
Formatted Answer
```

## 🔄 Dialogue Management

A lightweight dialogue frame is used to maintain conversational state.

A representative frame is:

```json
{
  "intent": "query_structured_stats",
  "slots": {
    "player": "Virat Kohli",
    "team": "India"
  },
  "route": "KB Pipeline"
}
```

Supported high-level intents include:

- `greeting`
- `query_structured_stats`
- `query_unstructured_text`

## 📊 Evaluation

The benchmark evaluates the system using three primary measures.

### Exact Match (EM)

Checks whether the normalized prediction exactly matches the expected answer.

```text
EM = Correct Exact Matches / Total Questions
```

### F1-Score

Measures token overlap between the predicted answer and the ground-truth answer using token-level precision and recall.

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

### Average Latency

Measures average processing time across the benchmark questions.

```text
Average Latency = Sum of Query Latencies / Number of Queries
```

Run the evaluation from the **Evaluation Summary** tab after launching the Streamlit application.

## 🧪 Example Questions

### Structured KB

```text
What is the NRR of India?
How many wickets did Mohammed Shami take?
How many runs did Virat Kohli score?
```

### IR / Factoid QA

```text
What was the pitch condition in the final?
Who was awarded Man of the Match in the final?
```

### Dialogue

```text
Hi
```

The chatbot responds with a greeting and establishes the dialogue interaction.

## ⚠️ Known Limitations

- TF-IDF relies on lexical overlap and can struggle with semantically similar questions using different vocabulary.
- Keyword-based routing is lightweight but can misclassify ambiguous queries.
- The local Transformer model requires additional memory and inference time compared with a purely rule-based system.
- The structured knowledge base only answers questions for information represented in the CSV files.
- The project is designed for the provided cricket corpus and benchmark rather than open-domain QA.

## 🔮 Future Improvements

Possible extensions include:

- Replace keyword routing with a trained intent classifier.
- Add semantic retrieval using sentence embeddings.
- Expand the cricket knowledge base.
- Add conversational context resolution for follow-up questions.
- Add answer-grounding and source citation to responses.
- Expand the benchmark with more diverse and adversarial questions.

## 📚 Deliverables Covered

This repository supports the major experiment deliverables:

- ✅ Working QA / chatbot prototype
- ✅ IR-based relevant passage retrieval
- ✅ Factoid answer extraction
- ✅ Structured knowledge-base querying
- ✅ Simple dialogue management
- ✅ Sample question-answer dataset
- ✅ Dialogue flow / frame design
- ✅ Evaluation using EM, F1, and latency
- ✅ Streamlit user interface

## 📌 Repository Information

**GitHub Repository URL:**  
https://github.com/donitjiju745/Cricket-Bot

**Repository Visibility:**  
**Public**

**Submission Branch:**  
`main`

**Current repository contents include:** `app.py`, `logic.py`, `requirements.txt`, `readme.md`, `.gitignore`, and the `data/` directory. citeturn733354view0

## 👨‍💻 Project

**Project:** Cricket QA & Dialogue Management System  
**Domain:** Cricket Question Answering and Chatbot  
**Architecture:** Hybrid IR + Extractive QA + Structured KB + Dialogue Management  
**Interface:** Streamlit  
**Execution:** Local / API-independent

## 📄 License

This project is intended for academic/educational use as part of the Question Answering and chatbot experiment.

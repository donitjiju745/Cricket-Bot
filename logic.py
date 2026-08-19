import os
import re
import string
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. SETUP: PATHS & OFFLINE MODEL LOADING
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "minilm")
CORPUS_DIR = os.path.join(BASE_DIR, "data", "corpus")

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(LOCAL_MODEL_PATH)

# ---------------------------------------------------------
# 2. IR SYSTEM: TF-IDF PASSAGE RETRIEVAL + EXTRACTIVE QA
# ---------------------------------------------------------
def load_corpus():
    """Loads all text files from the corpus directory."""
    documents = []
    for file_name in sorted(os.listdir(CORPUS_DIR)):
        if file_name.endswith(".txt"):
            file_path = os.path.join(CORPUS_DIR, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append({"title": file_name, "content": f.read().strip()})
    return documents

def retrieve_top_passages(query, top_k=2):
    """Retrieves top-K relevant passages using TF-IDF cosine similarity."""
    docs = load_corpus()
    corpus_texts = [d["content"] for d in docs]
    
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus_texts)
    query_vec = vectorizer.transform([query])
    
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    
    retrieved = []
    for idx in top_indices:
        if scores[idx] > 0.0:  # Only return passages with some relevance
            retrieved.append({
                "title": docs[idx]["title"],
                "content": docs[idx]["content"],
                "similarity_score": round(float(scores[idx]), 3)
            })
    
    # Fallback to the first document if no lexical match is found
    if not retrieved and docs:
        retrieved.append({
            "title": docs[0]["title"],
            "content": docs[0]["content"],
            "similarity_score": 0.0
        })
    return retrieved

def extract_factoid_answer(question, context):
    """Extracts answer span from context using the local PyTorch model."""
    start_time = time.time()
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Calculate softmax probabilities for confidence
    start_probs = torch.softmax(outputs.start_logits, dim=-1)
    end_probs = torch.softmax(outputs.end_logits, dim=-1)
    
    start_idx = torch.argmax(start_probs)
    end_idx = torch.argmax(end_probs) + 1
    
    # Extract string answer
    answer_tokens = inputs.input_ids[0][start_idx:end_idx]
    raw_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    
    # Confidence as joint probability percentage
    confidence = float(start_probs[0][start_idx] * end_probs[0][end_idx - 1]) * 100.0
    latency_ms = round((time.time() - start_time) * 1000, 2)
    
    # Post-clean extracted artifacts
    clean_answer = raw_answer.replace("##", "")
    if not clean_answer:
        clean_answer = "No exact answer span identified."
        confidence = 0.0
        
    return clean_answer, round(confidence, 2), latency_ms

# ---------------------------------------------------------
# 3. KNOWLEDGE BASE (KB) SYSTEM: STRUCTURED QUERIES
# ---------------------------------------------------------
def query_knowledge_base(query_text):
    """Parses entity and slot to retrieve exact data points from CSV files."""
    start_time = time.time()
    teams_df = pd.read_csv(os.path.join(BASE_DIR, "data", "teams_stats.csv"))
    players_df = pd.read_csv(os.path.join(BASE_DIR, "data", "players_stats.csv"))
    
    query_lower = query_text.lower()
    
    # Check Team Statistics
    for _, row in teams_df.iterrows():
        team_name = row["Team"].lower()
        if team_name in query_lower:
            if "nrr" in query_lower or "net run rate" in query_lower:
                return f"The Net Run Rate (NRR) of {row['Team']} is {row['NRR']}.", row.to_dict(), round((time.time() - start_time) * 1000, 2)
            if "point" in query_lower or "standing" in query_lower or "score" in query_lower:
                return f"{row['Team']} has played {row['Matches']} matches with {row['Points']} points (Won: {row['Won']}, Lost: {row['Lost']}).", row.to_dict(), round((time.time() - start_time) * 1000, 2)

    # Check Player Statistics
    for _, row in players_df.iterrows():
        player_name = row["Player"].lower()
        # Match full name or surname
        last_name = player_name.split()[-1]
        if player_name in query_lower or last_name in query_lower:
            if "wicket" in query_lower or "bowl" in query_lower:
                return f"{row['Player']} ({row['Team']}) has taken {row['Wickets']} wickets.", row.to_dict(), round((time.time() - start_time) * 1000, 2)
            if "run" in query_lower or "high score" in query_lower or "bat" in query_lower:
                return f"{row['Player']} ({row['Team']}) scored {row['Runs']} runs with a highest score of {row['HighScore']} and strike rate of {row['StrikeRate']}.", row.to_dict(), round((time.time() - start_time) * 1000, 2)
            return f"{row['Player']} is a {row['Role']} for {row['Team']} with {row['Runs']} runs and {row['Wickets']} wickets.", row.to_dict(), round((time.time() - start_time) * 1000, 2)

    latency_ms = round((time.time() - start_time) * 1000, 2)
    return "No matching record found in structured database.", {}, latency_ms

# ---------------------------------------------------------
# 4. DIALOGUE MANAGER & INTENT/SLOT FRAME TRACKER
# ---------------------------------------------------------
def dialogue_manager(user_message, current_frame=None):
    """
    Manages conversational frame state, classifies intent,
    fills slots, and routes to KB or IR.
    """
    if current_frame is None:
        current_frame = {
            "intent": "unknown",
            "slots": {"entity": None, "metric": None},
            "route": None
        }
    
    msg = user_message.lower().strip()
    
    # 1. Chitchat Intent
    if any(greet in msg for greet in ["hi", "hello", "hey", "namaste", "good morning", "good evening"]):
        current_frame["intent"] = "greeting"
        current_frame["route"] = "chitchat"
        return "Hello! I am your Cricket Assistant. Ask me about match summaries or player/team statistics.", current_frame, 100.0, ""
    
    if any(bye in msg for bye in ["bye", "thank you", "thanks", "exit"]):
        current_frame["intent"] = "goodbye"
        current_frame["route"] = "chitchat"
        return "Glad to be of assistance! Feel free to ask whenever you need more cricket insights.", current_frame, 100.0, ""

    # 2. Statistical / KB Intent
    kb_keywords = ["nrr", "net run rate", "points", "standing", "wicket", "wickets", "runs", "strike rate", "matches"]
    if any(kw in msg for kw in kb_keywords):
        current_frame["intent"] = "query_structured_stats"
        current_frame["route"] = "KB Pipeline"
        answer, record, _ = query_knowledge_base(user_message)
        current_frame["slots"]["data_record"] = record
        return answer, current_frame, 100.0, "Structured Database (CSV)"

    # 3. Informational / IR Intent
    current_frame["intent"] = "query_unstructured_text"
    current_frame["route"] = "IR Pipeline"
    passages = retrieve_top_passages(user_message, top_k=2)
    combined_context = " ".join([p["content"] for p in passages])
    answer, confidence, _ = extract_factoid_answer(user_message, combined_context)
    return answer, current_frame, confidence, f"Retrieved from: {', '.join([p['title'] for p in passages])}"

# ---------------------------------------------------------
# 5. EVALUATION METRICS (Exact Match, F1 Score)
# ---------------------------------------------------------
def normalize_text(s):
    """Lowercases text, strips punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_exact_match(prediction, ground_truth):
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    
    if not pred_tokens or not truth_tokens:
        return 1.0 if pred_tokens == truth_tokens else 0.0
        
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
        
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return round(2 * (precision * recall) / (precision + recall), 3)

def run_evaluation_benchmark():
    """Evaluates the full pipeline against sample_qa.csv."""
    qa_path = os.path.join(BASE_DIR, "data", "sample_qa.csv")
    df = pd.read_csv(qa_path)
    
    results = []
    for _, row in df.iterrows():
        q = row["Question"]
        gold = str(row["Expected_Answer"])
        q_type = row["Type"]
        
        start = time.time()
        if q_type == "KB":
            pred, _, _ = query_knowledge_base(q)
            # Check substring containment for conversational KB responses
            em = 1.0 if normalize_text(gold) in normalize_text(pred) else 0.0
            f1 = compute_f1(pred, gold)
        else:
            passages = retrieve_top_passages(q, top_k=2)
            context = " ".join([p["content"] for p in passages])
            pred, _, _ = extract_factoid_answer(q, context)
            em = compute_exact_match(pred, gold)
            f1 = compute_f1(pred, gold)
            
        latency = round((time.time() - start) * 1000, 2)
        results.append({
            "Question": q,
            "Type": q_type,
            "Expected": gold,
            "Predicted": pred,
            "Exact Match": em,
            "F1 Score": f1,
            "Latency (ms)": latency
        })
    return pd.DataFrame(results)
import streamlit as st
import pandas as pd
import logic

# ---------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cricket QA & Dialogue System",
    page_icon="🏏",
    layout="wide"
)

st.markdown("""
<style>
    /* Gradient Headers */
    .app-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #1E88E5, #D81B60);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .badge {
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-ir { background-color: #E3F2FD; color: #1565C0; }
    .badge-kb { background-color: #FCE4EC; color: #C2185B; }
    .frame-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 12px;
        border-radius: 4px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='app-header'>🏏 Hybrid Cricket QA & Dialogue Management System</div>", unsafe_allow_html=True)
st.caption("Extractive Information Retrieval (MiniLM) + Structured Knowledge Base + Dialogue Frame Tracking")

# ---------------------------------------------------------
# NAVIGATION TABS (All Project Deliverables)
# ---------------------------------------------------------
tabs = st.tabs([
    "💬 Chatbot Prototype",
    "🔎 IR Passage & Factoid QA",
    "📊 Structured KB Query",
    "📋 Sample QA Dataset",
    "🔄 Dialogue Flow & Frame Design",
    "📈 Evaluation Summary"
])

# ---------------------------------------------------------
# TAB 1: WORKING CHATBOT PROTOTYPE
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Working QA & Chatbot Prototype")
    st.write("Demonstrates live routing between conversational pleasantries, structured statistical lookups, and unstructured text retrieval.")

    col1, col2 = st.columns([2, 1])

    # Initialise session state for chat history and frame
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "dialogue_frame" not in st.session_state:
        st.session_state.dialogue_frame = {"intent": "None", "slots": {}, "route": "None"}

    with col1:
        # Display chat conversation
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "meta" in msg and msg["meta"]:
                    st.caption(msg["meta"])

        user_input = st.chat_input("Ask a question (e.g., 'What is India's NRR?' or 'Who was Man of the Match in the final?')")
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # Process through Dialogue Manager
            bot_reply, updated_frame, confidence, source = logic.dialogue_manager(
                user_input, st.session_state.dialogue_frame
            )
            st.session_state.dialogue_frame = updated_frame

            meta_info = f"Pipeline: {updated_frame['route']} | Confidence: {confidence}% | {source}"
            st.session_state.messages.append({"role": "assistant", "content": bot_reply, "meta": meta_info})
            
            with st.chat_message("assistant"):
                st.write(bot_reply)
                st.caption(meta_info)

    with col2:
        st.markdown("### 🛠 Active Dialogue Frame")
        st.write("Real-time conversational state maintained by the dialogue manager:")
        st.json(st.session_state.dialogue_frame)
        if st.button("Clear Chat & Reset State"):
            st.session_state.messages = []
            st.session_state.dialogue_frame = {"intent": "None", "slots": {}, "route": "None"}
            st.rerun()

# ---------------------------------------------------------
# TAB 2: IR PASSAGE & FACTOID QA ENDPOINT
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("Information Retrieval (IR) & Extractive Factoid QA")
    st.write("Demonstrates multi-document passage retrieval via TF-IDF cosine similarity followed by exact span extraction via MiniLM.")

    ir_query = st.text_input(
        "Enter your descriptive/factoid question:",
        value="What was the pitch condition during the final match in Ahmedabad?",
        key="ir_endpoint_input"
    )

    if st.button("Execute IR Pipeline", type="primary"):
        with st.spinner("Retrieving relevant passages and extracting answer..."):
            retrieved_docs = logic.retrieve_top_passages(ir_query, top_k=2)
            combined_text = " ".join([d["content"] for d in retrieved_docs])
            answer, conf, latency = logic.extract_factoid_answer(ir_query, combined_text)

            res_col1, res_col2 = st.columns([3, 1])
            with res_col1:
                st.success(f"**Extracted Factoid Answer:** {answer}")
            with res_col2:
                st.metric("Model Confidence", f"{conf}%", delta=f"{latency} ms")

            st.markdown("#### Retrieved Supporting Passages (TF-IDF Ranked):")
            for doc in retrieved_docs:
                with st.expander(f"📄 {doc['title']} (Similarity Score: {doc['similarity_score']})"):
                    st.write(doc["content"])

# ---------------------------------------------------------
# TAB 3: STRUCTURED KNOWLEDGE BASE (KB) ENDPOINT
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("Structured Knowledge Base Querying")
    st.write("Demonstrates deterministic entity and slot extraction mapped against relational tables.")

    kb_query = st.text_input(
        "Enter a statistical query for a player or team:",
        value="How many wickets has Mohammed Shami taken?",
        key="kb_endpoint_input"
    )

    if st.button("Query Knowledge Base", type="primary"):
        answer, record, latency = logic.query_knowledge_base(kb_query)
        
        kcol1, kcol2 = st.columns([3, 1])
        with kcol1:
            st.info(f"**Database Answer:** {answer}")
        with kcol2:
            st.metric("Execution Latency", f"{latency} ms")

        if record:
            st.markdown("#### Retrieved Record (Structured Tuple):")
            st.dataframe(pd.DataFrame([record]))

    st.divider()
    st.markdown("#### Browse Structured Knowledge Tables")
    view_tab1, view_tab2 = st.tabs(["Teams Standing & NRR", "Player Performance Stats"])
    with view_tab1:
        st.dataframe(pd.read_csv("data/teams_stats.csv"), use_container_width=True)
    with view_tab2:
        st.dataframe(pd.read_csv("data/players_stats.csv"), use_container_width=True)

# ---------------------------------------------------------
# TAB 4: SAMPLE QA DATASET
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("Sample Question-Answer Benchmark Set")
    st.write("Pre-annotated questions and ground truth labels used for system evaluation.")
    
    qa_df = pd.read_csv("data/sample_qa.csv")
    st.dataframe(qa_df, use_container_width=True)

# ---------------------------------------------------------
# TAB 5: DIALOGUE FLOW & FRAME DESIGN
# ---------------------------------------------------------
with tabs[4]:
    st.subheader("Dialogue Flow Architecture & Frame Specification")
    st.write("Overview of the hybrid routing logic and conversational frame management schema:")

    st.markdown("""
```text
  [ User Query ]
        │
        ▼
  [ Intent Classifier ] ────────────────────────────────────────┐
        │                                                       │
 ┌──────┴──────────────────────┬──────────────────────┐         │
 ▼                             ▼                      ▼         ▼
[ Greeting / Chitchat ]   [ Statistical Query ]  [ Factoid Q ]  [ Fallback ]
 │                             │                      │         │
 ├─► Direct Conversational     ├─► Extract Entities   ├─► TF-IDF Passage
 │   Template Reply            │   (Team, Player)     │   Retrieval
 │                             ├─► Query SQL/CSV      ├─► Extractive Span
 │                             │   Data Base          │   Inference (MiniLM)
 │                             ▼                      ▼         │
 └────────────────────────► [ Frame State Update ] ◄────────────┘
                               │
                               ▼
                    [ Formatted UI Output ]
""")

st.markdown("#### Frame Structure Specification")
st.code("""
{
"session_id": "uuid4_string",
"turn_count": 3,
"intent": "query_player_wickets",
"slots": {
"player_name": "Mohammed Shami",
"team": "India",
"metric_requested": "wickets",
"resolved_value": 24
},
"route": "KB Pipeline",
"dialogue_status": "COMPLETED"
}
""", language="json")

#---------------------------------------------------------
#TAB 6: EVALUATION SUMMARY
#---------------------------------------------------------
with tabs[5]:
    st.subheader("System Evaluation Summary & Benchmark Results")
    st.write("Quantitative assessment measuring Exact Match (EM), Token-level F1-Score, and Latency across both pipelines.")

if st.button("Run System Evaluation Benchmark", type="primary"):
    with st.spinner("Evaluating sample benchmark set..."):
        eval_results = logic.run_evaluation_benchmark()
        
        # Aggregate metrics
        avg_em = round(eval_results["Exact Match"].mean() * 100, 1)
        avg_f1 = round(eval_results["F1 Score"].mean() * 100, 1)
        avg_lat = round(eval_results["Latency (ms)"].mean(), 2)

        m1, m2, m3 = st.columns(3)
        m1.metric("Average Exact Match (EM)", f"{avg_em}%")
        m2.metric("Average F1-Score", f"{avg_f1}%")
        m3.metric("Average Pipeline Latency", f"{avg_lat} ms")

        st.markdown("#### Detailed Benchmark Test Results:")
        st.dataframe(eval_results, use_container_width=True)
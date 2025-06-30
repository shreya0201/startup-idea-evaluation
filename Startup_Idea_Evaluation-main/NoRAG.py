# app.py

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pytrends.request import TrendReq
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np

# -----------------------------------------------------------------------------
# 1) Environment & Caching
# -----------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_HOME"] = "D:/hf_cache"
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# 2) Load Crunchbase Data & Precompute Category Stats
# -----------------------------------------------------------------------------
@st.cache_data
def load_crunchbase_stats(path="data/crunchbase.csv"):
    df = pd.read_csv(path)
    df["funding_total_usd"] = pd.to_numeric(df["funding_total_usd"], errors="coerce")

    conditions = [
        df["status"].str.lower() == "operating",
        df["status"].str.lower() == "acquired", 
        df["status"].str.lower() == "ipo"
    ]
    df["is_successful"] = np.select(conditions, [1, 0.8, 1], default=0)

    cat_stats = (
        df[df['category_list'].notna()]
          .groupby("category_list")
          .agg(
            startup_count=("name", "count"),
            avg_funding=("funding_total_usd", "mean"),
            avg_rounds=("funding_rounds", "mean"),
            success_rate=("is_successful", "mean")
          )
    )

    cat_stats["success_rate"] = (cat_stats["success_rate"] * 100).round(1)
    cat_stats["avg_funding"] = cat_stats["avg_funding"].fillna(0).round(0)
    cat_stats["avg_rounds"] = cat_stats["avg_rounds"].fillna(0).round(1)
    cat_stats = cat_stats[cat_stats["startup_count"] >= 3]
    return df, cat_stats

df, category_stats = load_crunchbase_stats()

# -----------------------------------------------------------------------------
# 3) Load Phi-2 Model Locally
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_generator(model_name="microsoft/phi-2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return gen_pipe

generator = load_generator()

# -----------------------------------------------------------------------------
# 4) Enhanced Analysis Functions
# -----------------------------------------------------------------------------
def extract_keywords_with_phi2(idea: str) -> list:
    prompt = f"""
    Below are some startup ideas and their relevant keywords. Follow the pattern and extract 3–6 concise, comma-separated keywords from the final idea.

    Startup Idea: An AI-powered mobile app that diagnoses plant diseases for farmers.
    Keywords: agriculture, plant health, AI, mobile app, crop diagnostics

    Startup Idea: A subscription box delivering eco-friendly, zero-waste personal care products.
    Keywords: sustainability, personal care, zero-waste, subscription, eco-friendly

    Startup Idea: An online platform connecting remote tutors with school students.
    Keywords: edtech, online learning, remote tutoring, students, education

    Startup Idea: A marketplace for renting luxury fashion instead of buying.
    Keywords: fashion tech, rental, luxury fashion, sustainability, reuse

    Startup Idea: {idea}
    Keywords:
    """
    out = generator(prompt, max_new_tokens=40, do_sample=False)[0]["generated_text"]
    part = out.split("Keywords:")[-1].strip().split("\n")[0]

    if ',' in part:
        kws = [kw.strip().lower() for kw in part.split(",") if kw.strip()]
    else:
        kws = re.findall(r'\b\w+\b', part.lower())

    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
    kws = [kw for kw in kws if kw not in stop_words and len(kw) > 2][:6]

    # ✅ Fallback mechanism
    if not kws:
        fallback_pool = ['AI', 'startup', 'app', 'tech', 'platform', 'solution', 'sustainability', 'eco-friendly', 'fashion']
        for word in fallback_pool:
            if word.lower() in idea.lower():
                kws.append(word.lower())
        kws = list(set(kws))[:6]

    return kws

def analyze_startup_idea_trends(keywords: list) -> dict:
    if not keywords:
        return {}
    pytrends = TrendReq()
    scores = {}
    for kw in keywords:
        try:
            pytrends.build_payload([kw], timeframe='today 12-m')
            df_tr = pytrends.interest_over_time()
            if not df_tr.empty and kw in df_tr.columns:
                recent_avg = df_tr[kw].tail(12).mean()
                scores[kw] = recent_avg
            else:
                scores[kw] = 0
        except Exception:
            scores[kw] = 0
    return scores

def match_keywords_to_categories(keywords: list):
    matches = []
    match_scores = []
    for kw in keywords:
        best_matches = []
        for cat in category_stats.index:
            if kw in cat.lower():
                best_matches.append((cat, category_stats.loc[cat], 1.0))
            elif any(word in cat.lower() for word in kw.split()):
                best_matches.append((cat, category_stats.loc[cat], 0.7))
            elif any(kw_word in cat.lower() for kw_word in kw.split() if len(kw_word) > 3):
                best_matches.append((cat, category_stats.loc[cat], 0.5))
        for match in best_matches[:3]:
            matches.append(match)
            match_scores.append(match[2])
    seen = set()
    final_matches = []
    final_scores = []
    for i, (cat, row, score) in enumerate(matches):
        if cat not in seen:
            seen.add(cat)
            final_matches.append((cat, row))
            final_scores.append(score)
    return final_matches, final_scores

def calculate_final_metrics(matches, match_scores, trend_scores, keywords):
    if not matches:
        return None, None, 0
    funding_values = []
    success_values = []
    round_values = []
    weights = []
    for i, (cat, stats) in enumerate(matches):
        if not pd.isna(stats['avg_funding']) and not pd.isna(stats['success_rate']):
            funding_values.append(stats['avg_funding'])
            success_values.append(stats['success_rate'])
            round_values.append(stats['avg_rounds'])
            base_weight = match_scores[i] if i < len(match_scores) else 0.5
            trend_weight = 0
            for kw in keywords:
                if kw in cat.lower():
                    trend_weight += trend_scores.get(kw, 0) / 100.0
            combined_weight = base_weight + min(trend_weight, 0.5)
            weights.append(combined_weight)
    if not funding_values:
        return None, None, 0
    total_weight = sum(weights)
    if total_weight > 0:
        weighted_funding = sum(f * w for f, w in zip(funding_values, weights)) / total_weight
        weighted_success = sum(s * w for s, w in zip(success_values, weights)) / total_weight
        confidence = min(len(matches) * 20, 100)
    else:
        weighted_funding = np.mean(funding_values)
        weighted_success = np.mean(success_values)
        confidence = 30
    return int(weighted_funding), round(weighted_success, 1), confidence

# Your semantic matcher and UI logic follow here unchanged...


# 📦 Load Sentence-Transformer for semantic matching

from sentence_transformers import SentenceTransformer, util
import torch

@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

semantic_model = load_semantic_model()

# 🧠 Optional: Semantic similarity-based keyword-to-category matcher
def semantic_match_keywords_to_categories(keywords, top_k=5, similarity_threshold=0.4):
    cat_names = list(category_stats.index)
    
    # Precompute all Crunchbase category embeddings
    cat_embeddings = semantic_model.encode(cat_names, convert_to_tensor=True)
    
    matched = []
    for kw in keywords:
        kw_embedding = semantic_model.encode(kw, convert_to_tensor=True)
        scores = util.cos_sim(kw_embedding, cat_embeddings)[0]
        
        # Get top matches above threshold
        top_results = [(cat_names[i], float(scores[i])) for i in scores.argsort(descending=True)]
        for cat, sim in top_results[:top_k]:
            if sim >= similarity_threshold:
                matched.append((cat, sim))

    # Deduplicate and retain best match for each category
    best_matches = {}
    for cat, score in matched:
        if cat not in best_matches or best_matches[cat] < score:
            best_matches[cat] = score

    final_matches = [(cat, category_stats.loc[cat]) for cat in best_matches]
    final_scores  = [best_matches[cat] for cat in best_matches]
    return final_matches, final_scores


USE_SEMANTIC_MATCHING = True  # Set to False to go back to your original matcher


# -----------------------------------------------------------------------------
# 5) Streamlit UI
# -----------------------------------------------------------------------------
st.title("🚀 Startup Idea Evaluator")
st.markdown("*Get data-driven insights on your startup idea's potential success and funding requirements*")

idea = st.text_area("💡 Enter your startup idea here:", 
                   placeholder="e.g., A mobile app that uses AI to help people find sustainable fashion alternatives...",
                   height=120)

if st.button("🔍 Evaluate Idea", type="primary"):
    if not idea.strip():
        st.warning("Please enter a startup idea to evaluate.")
    else:
        with st.spinner("Analyzing your startup idea..."):
            # 1) Extract Keywords
            keywords = extract_keywords_with_phi2(idea)
            
            if not keywords:
                st.error("Could not extract meaningful keywords from your idea. Please try rephrasing it.")
                st.stop()
            
            # 2) Analyze Trends (optional display)
            trend_scores = analyze_startup_idea_trends(keywords)
            
            # 3) Match to Categories
            if USE_SEMANTIC_MATCHING:
                matches, match_scores = semantic_match_keywords_to_categories(keywords)
            else:
                matches, match_scores = match_keywords_to_categories(keywords)
            
            # 4) Calculate Final Metrics
            final_funding, final_success, confidence = calculate_final_metrics(
                matches, match_scores, trend_scores, keywords
            )
            
            # 5) Display Results
            st.markdown("---")
            st.subheader("📊 Evaluation Results")
            
            if final_funding is not None and final_success is not None:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="💰 Estimated Funding Needed",
                        value=f"${final_funding:,}",
                        help="Average funding raised by similar startups"
                    )
                
                with col2:
                    st.metric(
                        label="✅ Success Rate",
                        value=f"{final_success}%",
                        help="Percentage of similar startups that remained operational or got acquired/IPO"
                    )
                
                with col3:
                    st.metric(
                        label="🎯 Confidence",
                        value=f"{confidence}%",
                        help="Reliability of the analysis based on available data"
                    )
                
                # Additional insights
                if confidence >= 60:
                    if final_success >= 40:
                        st.success("🎉 Your idea shows strong potential based on historical data!")
                    elif final_success >= 25:
                        st.info("💡 Moderate potential - consider refining your approach or market positioning.")
                    else:
                        st.warning("⚠️ Lower success rate in this category - ensure you have a strong differentiation strategy.")
                else:
                    st.info("📊 Limited historical data available. Consider this a preliminary assessment.")
                
                # Show keywords used for analysis
                with st.expander("🔍 Analysis Details"):
                    st.write("**Keywords extracted:**", ", ".join(keywords))
                    st.write(f"**Categories analyzed:** {len(matches)} relevant categories found")
                    if trend_scores:
                        avg_trend = np.mean(list(trend_scores.values()))
                        st.write(f"**Market interest:** {avg_trend:.1f}/100 (Google Trends)")
                
            else:
                st.warning("❌ No relevant historical data found for your startup idea.")
                st.info("This could mean your idea is very innovative, or the keywords need refinement.")
                
                if keywords:
                    st.write("**Keywords extracted:**", ", ".join(keywords))
                    st.write("Try rephrasing your idea with more specific industry terms or product categories.")

# Footer
st.markdown("---")
st.markdown("*Based on Crunchbase data and Google Trends analysis*")

#to start 
#.\venv\Scripts\activate
#pip install -r requirements.txt
#if streamlit not found then pip install streamlit
#streamlit run app.py

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
import json
from typing import List, Dict, Any, Tuple
import pickle
from pathlib import Path

# -----------------------------------------------------------------------------
# 1) Environment & Caching
# -----------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_HOME"] = "D:/hf_cache"
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# 2) Funding Display Helper Functions
# -----------------------------------------------------------------------------
def format_funding_display(amount):
    """Format funding amount for display with M suffix for millions"""
    if amount >= 1000000:
        return f"${amount/1000000:.1f}M"
    elif amount >= 1000:
        return f"${amount/1000:.1f}K"
    else:
        return f"${amount:,.0f}"

def get_exact_funding_text(amount):
    """Get exact funding amount for hover tooltip"""
    return f"Exact amount: ${amount:,.0f}"

# -----------------------------------------------------------------------------
# 3) Vector Database Fallback Logic
# -----------------------------------------------------------------------------
try:
    import faiss
    VECTOR_DB = "faiss"
    print("✅ Using FAISS for vector storage")
except ImportError:
    try:
        import chromadb
        VECTOR_DB = "chroma"
        print("✅ Using ChromaDB for vector storage")
    except ImportError:
        st.error("Neither FAISS nor ChromaDB is available. Please install one of them.")
        st.stop()

# -----------------------------------------------------------------------------
# 4) Load Sentence Transformer for embeddings
# -----------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer, util
import torch

@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

semantic_model = load_semantic_model()

# -----------------------------------------------------------------------------
# 5) RAG Vector Store Implementation
# -----------------------------------------------------------------------------
class RAGVectorStore:
    def __init__(self, embedding_model, vector_db_type="faiss"):
        self.embedding_model = embedding_model
        self.vector_db_type = vector_db_type
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        self.index = None
        self.documents = []
        self.metadata = []
        self.chroma_client = None
        self.chroma_collection = None
        
    def _init_faiss(self):
        """Initialize FAISS index"""
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
    def _init_chroma(self):
        """Initialize ChromaDB"""
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        try:
            self.chroma_collection = self.chroma_client.get_collection("startup_data")
        except:
            self.chroma_collection = self.chroma_client.create_collection("startup_data")
    
    def add_documents(self, documents: List[str], metadata: List[Dict]):
        """Add documents to vector store"""
        if self.vector_db_type == "faiss":
            if self.index is None:
                self._init_faiss()
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents, convert_to_tensor=False)
            embeddings = np.array(embeddings).astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            self.index.add(embeddings)
            self.documents.extend(documents)
            self.metadata.extend(metadata)
            
        elif self.vector_db_type == "chroma":
            if self.chroma_collection is None:
                self._init_chroma()
            
            # Add to ChromaDB
            ids = [f"doc_{len(self.documents) + i}" for i in range(len(documents))]
            self.chroma_collection.add(
                documents=documents,
                metadatas=metadata,
                ids=ids
            )
            self.documents.extend(documents)
            self.metadata.extend(metadata)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        """Search for similar documents"""
        if not self.documents:
            return []
            
        if self.vector_db_type == "faiss":
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
            query_embedding = np.array(query_embedding).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:  # Valid index
                    results.append((self.documents[idx], self.metadata[idx], float(score)))
            return results
            
        elif self.vector_db_type == "chroma":
            # Search using ChromaDB
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=min(top_k, len(self.documents))
            )
            
            search_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                # Convert distance to similarity score (ChromaDB uses cosine distance)
                similarity = 1 - distance
                search_results.append((doc, metadata, similarity))
            
            return search_results
        
        return []

# -----------------------------------------------------------------------------
# 6) Load Crunchbase Data & Build RAG Index
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

@st.cache_resource
def build_rag_index(_df, _semantic_model):
    """Build RAG index from Crunchbase data"""
    vector_store = RAGVectorStore(_semantic_model, VECTOR_DB)
    
    documents = []
    metadata = []
    
    # Create documents from startup data
    for _, row in _df.iterrows():
        if pd.notna(row.get('name')) and pd.notna(row.get('category_list')):
            # Create document text
            doc_text = f"Startup: {row['name']}\n"
            doc_text += f"Category: {row['category_list']}\n"
            
            if pd.notna(row.get('short_description')):
                doc_text += f"Description: {row['short_description']}\n"
            
            doc_text += f"Status: {row.get('status', 'Unknown')}\n"
            doc_text += f"Funding Total: ${row.get('funding_total_usd', 0):,.0f}\n"
            doc_text += f"Funding Rounds: {row.get('funding_rounds', 0)}\n"
            
            if pd.notna(row.get('founded_at')):
                doc_text += f"Founded: {row['founded_at']}\n"
            
            if pd.notna(row.get('country_code')):
                doc_text += f"Country: {row['country_code']}\n"
            
            documents.append(doc_text)
            
            # Create metadata
            meta = {
                'name': row['name'],
                'category': row['category_list'],
                'status': row.get('status', 'Unknown'),
                'funding_total': float(row.get('funding_total_usd', 0)),
                'funding_rounds': int(row.get('funding_rounds', 0)),
                'founded_at': str(row.get('founded_at', '')),
                'country': str(row.get('country_code', '')),
                'description': str(row.get('short_description', ''))
            }
            metadata.append(meta)
    
    # Add documents to vector store in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadata[i:i+batch_size]
        vector_store.add_documents(batch_docs, batch_meta)
    
    return vector_store

df, category_stats = load_crunchbase_stats()

# Build RAG index
print("Building RAG index...")
rag_store = build_rag_index(df, semantic_model)
print(f"RAG index built with {len(rag_store.documents)} documents")

# -----------------------------------------------------------------------------
# 7) Load Phi-2 Model Locally
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_generator(model_name="microsoft/phi-2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return gen_pipe

generator = load_generator()

# -----------------------------------------------------------------------------
# 8) RAG-Enhanced Analysis Functions
# -----------------------------------------------------------------------------
def extract_keywords_with_rag(idea: str, rag_store: RAGVectorStore) -> list:
    """Extract keywords using both Phi-2 and RAG context"""
    
    # First, get relevant context from RAG
    similar_docs = rag_store.search(idea, top_k=3)
    
    # Build context from similar startups
    context = ""
    if similar_docs:
        context = "Similar successful startups:\n"
        for doc, meta, score in similar_docs[:2]:
            context += f"- {meta['name']} ({meta['category']}): {meta['description'][:100]}...\n"
        context += "\n"
    
    prompt = f"""
    {context}Based on the above context and similar startups, extract 3–6 concise, comma-separated keywords from the startup idea below.

    Examples:
    Startup Idea: An AI-powered mobile app that diagnoses plant diseases for farmers.
    Keywords: agriculture, plant health, AI, mobile app, crop diagnostics

    Startup Idea: A subscription box delivering eco-friendly, zero-waste personal care products.
    Keywords: sustainability, personal care, zero-waste, subscription, eco-friendly

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

    # Enhanced fallback with RAG context
    if not kws and similar_docs:
        # Extract keywords from similar startup categories
        for doc, meta, score in similar_docs:
            category_words = meta['category'].lower().split(',')
            for word in category_words:
                clean_word = word.strip()
                if len(clean_word) > 2 and clean_word not in stop_words:
                    kws.append(clean_word)
        kws = list(set(kws))[:6]
    
    # Original fallback
    if not kws:
        fallback_pool = ['AI', 'startup', 'app', 'tech', 'platform', 'solution', 'sustainability', 'eco-friendly', 'fashion']
        for word in fallback_pool:
            if word.lower() in idea.lower():
                kws.append(word.lower())
        kws = list(set(kws))[:6]

    return kws

def generate_rag_enhanced_insights(idea: str, keywords: list, rag_store: RAGVectorStore) -> Dict[str, Any]:
    """Generate insights using RAG-retrieved data"""
    
    # Search for similar startups
    similar_docs = rag_store.search(idea, top_k=10)
    
    if not similar_docs:
        return {"error": "No similar startups found in database"}
    
    # Analyze retrieved data
    funding_amounts = []
    success_examples = []
    failed_examples = []
    categories = set()
    countries = set()
    
    for doc, meta, score in similar_docs:
        funding_amounts.append(meta['funding_total'])
        categories.add(meta['category'])
        if meta['country']:
            countries.add(meta['country'])
        
        if meta['status'].lower() in ['operating', 'acquired', 'ipo']:
            success_examples.append(meta)
        else:
            failed_examples.append(meta)
    
    # Calculate metrics
    avg_funding = np.mean([f for f in funding_amounts if f > 0]) if funding_amounts else 0
    median_funding = np.median([f for f in funding_amounts if f > 0]) if funding_amounts else 0
    success_rate = len(success_examples) / len(similar_docs) * 100 if similar_docs else 0
    
    # Build context for LLM
    context = f"""
    Analysis of {len(similar_docs)} similar startups:
    - Average funding: ${avg_funding:,.0f}
    - Median funding: ${median_funding:,.0f}
    - Success rate: {success_rate:.1f}%
    - Main categories: {', '.join(list(categories)[:3])}
    - Key markets: {', '.join(list(countries)[:3])}
    
    Top successful examples:
    """
    
    for example in success_examples[:3]:
        context += f"- {example['name']}: ${example['funding_total']:,.0f} funding, Status: {example['status']}\n"
    
    return {
        "avg_funding": int(avg_funding),
        "median_funding": int(median_funding),
        "success_rate": round(success_rate, 1),
        "confidence": min(len(similar_docs) * 10, 100),
        "similar_count": len(similar_docs),
        "categories": list(categories),
        "success_examples": success_examples[:5],
        "context": context
    }

def generate_detailed_analysis(idea: str, keywords: list, rag_insights: Dict, rag_store: RAGVectorStore) -> str:
    """Generate detailed analysis using RAG context"""
    
    if "error" in rag_insights:
        return "Unable to generate detailed analysis due to insufficient data."
    
    # Get market trend data
    trend_scores = analyze_startup_idea_trends(keywords)
    avg_trend = np.mean(list(trend_scores.values())) if trend_scores else 0
    
    context = rag_insights.get("context", "")
    
    prompt = f"""
    {context}
    
    Based on the above data analysis of similar startups, provide a comprehensive evaluation of this startup idea:
    "{idea}"
    
    Keywords: {', '.join(keywords)}
    Market Interest Score: {avg_trend:.1f}/100
    
    Provide analysis covering:
    1. Market opportunity and positioning
    2. Funding expectations and strategy
    3. Key success factors based on similar companies
    4. Potential challenges and risks
    5. Actionable recommendations
    
    Analysis:
    """
    
    response = generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)[0]["generated_text"]
    analysis = response.split("Analysis:")[-1].strip()
    
    return analysis

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

# -----------------------------------------------------------------------------
# 9) Streamlit UI
# -----------------------------------------------------------------------------
st.title("🚀 RAG-Enhanced Startup Idea Evaluator")
st.markdown("*Get AI-powered insights based on real startup data from Crunchbase*")
st.markdown(f"*Using {VECTOR_DB.upper()} for vector storage with {len(rag_store.documents):,} startup records*")

idea = st.text_area("💡 Enter your startup idea here:", 
                   placeholder="e.g., A mobile app that uses AI to help people find sustainable fashion alternatives...",
                   height=120)

if st.button("🔍 Evaluate Idea", type="primary"):
    if not idea.strip():
        st.warning("Please enter a startup idea to evaluate.")
    else:
        with st.spinner("Analyzing your startup idea using AI and real data..."):
            
            # 1) Extract Keywords with RAG context
            keywords = extract_keywords_with_rag(idea, rag_store)
            
            if not keywords:
                st.error("Could not extract meaningful keywords from your idea. Please try rephrasing it.")
                st.stop()
            
            # 2) Get RAG-enhanced insights
            rag_insights = generate_rag_enhanced_insights(idea, keywords, rag_store)
            
            if "error" in rag_insights:
                st.warning("⚠️ Limited data available for analysis. Showing basic metrics.")
                
                # Fallback to original analysis
                trend_scores = analyze_startup_idea_trends(keywords)
                matches, match_scores = semantic_match_keywords_to_categories(keywords)
                final_funding, final_success, confidence = calculate_final_metrics(
                    matches, match_scores, trend_scores, keywords
                )
                
                if final_funding and final_success:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("💰 Est. Funding", format_funding_display(final_funding))
                    with col2:
                        st.metric("✅ Success Rate", f"{final_success}%")
                    with col3:
                        st.metric("🎯 Confidence", f"{confidence}%")
                
            else:
                # 3) Display RAG-enhanced results
                st.markdown("---")
                st.subheader("📊 AI-Enhanced Evaluation Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_funding = rag_insights['avg_funding']
                    st.metric(
                        label="💰 Average Funding",
                        value=format_funding_display(avg_funding),
                        help=get_exact_funding_text(avg_funding)
                    )
                
                with col2:
                    median_funding = rag_insights['median_funding']
                    st.metric(
                        label="📊 Median Funding",
                        value=format_funding_display(median_funding),
                        help=get_exact_funding_text(median_funding)
                    )
                
                with col3:
                    st.metric(
                        label="✅ Success Rate",
                        value=f"{rag_insights['success_rate']}%",
                        help="Percentage of similar startups still operating/acquired/IPO"
                    )
                
                with col4:
                    st.metric(
                        label="🎯 Confidence",
                        value=f"{rag_insights['confidence']}%",
                        help=f"Based on {rag_insights['similar_count']} similar startups"
                    )
                
                # Success indicators
                if rag_insights['confidence'] >= 60:
                    if rag_insights['success_rate'] >= 40:
                        st.success("🎉 Strong potential based on similar startup performance!")
                    elif rag_insights['success_rate'] >= 25:
                        st.info("💡 Moderate potential - focus on differentiation strategy.")
                    else:
                        st.warning("⚠️ Lower success rate - ensure strong competitive advantages.")
                else:
                    st.info("📊 Preliminary assessment based on available data.")
                
                # 4) Generate detailed AI analysis
                st.markdown("---")
                st.subheader("🤖 AI-Generated Analysis")
                
                with st.spinner("Generating detailed analysis..."):
                    detailed_analysis = generate_detailed_analysis(idea, keywords, rag_insights, rag_store)
                    st.write(detailed_analysis)
                
                # 5) Show similar startups
                if rag_insights.get('success_examples'):
                    with st.expander("📈 Similar Successful Startups"):
                        for example in rag_insights['success_examples']:
                            st.write(f"**{example['name']}** ({example['category']})")
                            funding_display = format_funding_display(example['funding_total'])
                            exact_funding = f"${example['funding_total']:,.0f}"
                            st.write(f"💰 Funding: {funding_display} (Exact: {exact_funding}) | Status: {example['status']}")
                            if example['description']:
                                st.write(f"📝 {example['description'][:200]}...")
                            st.write("---")
                
                # 6) Analysis details
                with st.expander("🔍 Analysis Details"):
                    st.write("**Keywords extracted:**", ", ".join(keywords))
                    st.write(f"**Similar startups analyzed:** {rag_insights['similar_count']}")
                    st.write("**Categories found:**", ", ".join(rag_insights['categories'][:5]))
                    
                    # Show trend data if available
                    trend_scores = analyze_startup_idea_trends(keywords)
                    if trend_scores:
                        avg_trend = np.mean(list(trend_scores.values()))
                        st.write(f"**Market interest (Google Trends):** {avg_trend:.1f}/100")

# -----------------------------------------------------------------------------
# 10) Enhanced helper functions (keeping the original ones for fallback)
# -----------------------------------------------------------------------------
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

# Footer
st.markdown("---")
st.markdown(f"*Powered by RAG with {VECTOR_DB.upper()}, Phi-2 LLM, and real Crunchbase data*")
st.markdown("*Analysis based on semantic similarity to historical startup performance*")

#to create new venv
#python -m venv venv
#to start 
#.\venv\Scripts\activate
#pip install -r requirements.txt
#if streamlit not found then pip install streamlit
#streamlit run app.py
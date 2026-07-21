# pyrefly: ignore [missing-import]
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import traceback

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    try:
        nltk.download('vader_lexicon', quiet=True)
    except Exception:
        pass

@st.cache_resource(show_spinner=False)
def load_hf_sentiment_pipeline():
    """
    Attempts to load the Hugging Face sentiment analysis pipeline.
    Returns (pipeline, success).
    """
    try:
        from transformers import pipeline
        # Use a small, standard sentiment analysis model
        classifier = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1 # Run on CPU
        )
        return classifier, True
    except Exception as e:
        # Fallback if transformers, torch, or download fail
        return None, False

def analyze_sentiments(df):
    """
    Analyzes reviews in the dataframe.
    Adds a 'Sentiment' column to the dataframe.
    Returns (processed_df, model_used).
    """
    df = df.copy()
    
    # Read settings from session state (defaults if not present)
    engine_choice = st.session_state.get("sentiment_engine", "Auto (Hugging Face preferred)")
    conf_threshold = st.session_state.get("confidence_threshold", 0.6)
    
    hf_classifier = None
    hf_success = False
    
    # Decide whether to load Hugging Face based on user choice
    if engine_choice in ["Auto (Hugging Face preferred)", "Advanced Transformer (DistilBERT only)"]:
        hf_classifier, hf_success = load_hf_sentiment_pipeline()
        
    # If forced Transformer but it failed, show a warning or fallback if Auto
    if engine_choice == "Advanced Transformer (DistilBERT only)" and not hf_success:
        st.warning("⚠️ Could not load Hugging Face pipeline. Falling back to VADER.")
        
    if (hf_success and hf_classifier is not None) and (engine_choice != "Fast NLP (VADER only)"):
        model_used = "Advanced AI (Transformer)"
        sentiments = []
        
        # Process in batch or individually
        for idx, row in df.iterrows():
            review = str(row["Review"]).strip()
            rating = int(row["Rating"])
            
            # Simple rule: if rating is 3, default to Neutral
            if rating == 3:
                sentiments.append("Neutral")
                continue
                
            try:
                # Truncate review to 512 characters to avoid token limit errors
                truncated_review = review[:512]
                res = hf_classifier(truncated_review)[0]
                label = res["label"].upper() # 'POSITIVE' or 'NEGATIVE'
                score = res["score"]
                
                # Check confidence threshold
                if label == "POSITIVE" and score > conf_threshold:
                    sentiments.append("Positive")
                elif label == "NEGATIVE" and score > conf_threshold:
                    sentiments.append("Negative")
                else:
                    # Low confidence defaults to rating-based fallback
                    if rating > 3:
                        sentiments.append("Positive")
                    elif rating < 3:
                        sentiments.append("Negative")
                    else:
                        sentiments.append("Neutral")
            except Exception:
                # Fallback to rating logic if specific review fails
                if rating > 3:
                    sentiments.append("Positive")
                elif rating < 3:
                    sentiments.append("Negative")
                else:
                    sentiments.append("Neutral")
                    
        df["Sentiment"] = sentiments
    else:
        # Fallback to NLTK VADER
        model_used = "Fast NLP Engine (VADER)"
        sentiments = []
        
        try:
            sia = SentimentIntensityAnalyzer()
            for idx, row in df.iterrows():
                review = str(row["Review"]).strip()
                rating = int(row["Rating"])
                
                # Analyze with VADER
                scores = sia.polarity_scores(review)
                compound = scores["compound"]
                
                # Hybrid VADER + Rating heuristic for high accuracy
                if compound >= 0.05:
                    # Confirmed Positive, but double check if rating is 1 or 2
                    if rating <= 2:
                        sentiments.append("Neutral") # Dissonance, tone is positive but rating low
                    else:
                        sentiments.append("Positive")
                elif compound <= -0.05:
                    # Confirmed Negative, but double check if rating is 4 or 5
                    if rating >= 4:
                        sentiments.append("Neutral") # Dissonance, tone is negative but rating high
                    else:
                        sentiments.append("Negative")
                else:
                    # Near zero compound -> Neutral. Double check rating
                    if rating >= 4:
                        sentiments.append("Positive")
                    elif rating <= 2:
                        sentiments.append("Negative")
                    else:
                        sentiments.append("Neutral")
        except Exception as e:
            # Bare fallback using Rating only if VADER initialization fails
            model_used = "Basic Heuristic Engine (Rating-based)"
            for idx, row in df.iterrows():
                rating = int(row["Rating"])
                if rating > 3:
                    sentiments.append("Positive")
                elif rating < 3:
                    sentiments.append("Negative")
                else:
                    sentiments.append("Neutral")
                    
        df["Sentiment"] = sentiments
        
    return df, model_used

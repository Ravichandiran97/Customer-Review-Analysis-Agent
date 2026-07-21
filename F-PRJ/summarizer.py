# pyrefly: ignore [missing-import]
import streamlit as st
import re
import math
from collections import Counter
import traceback

# Try to import NLTK for sentence tokenization
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception:
        pass

@st.cache_resource(show_spinner=False)
def load_hf_summarization_pipeline():
    """
    Attempts to load the Hugging Face summarization pipeline.
    Returns (pipeline, success).
    """
    try:
        from transformers import pipeline
        # Use t5-small as it is lightweight (~242MB) and fast
        summarizer = pipeline(
            "summarizer", 
            model="t5-small",
            device=-1 # Run on CPU
        )
        return summarizer, True
    except Exception as e:
        return None, False

def fallback_extractive_summary(reviews_list, num_sentences=3):
    """
    High-quality TF-IDF and word frequency based extractive summarizer.
    Extremely fast, reliable, and requires no external model downloads.
    """
    text = " ".join([str(r).strip() for r in reviews_list if str(r).strip()])
    if not text:
        return "No content to summarize."
        
    # Split text into sentences
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Regex fallback for sentence splitting
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
        
    # Simple Stopwords List
    stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
        'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
        'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
        'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
        "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
        'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
        'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
        'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'the', 'this', 'that', 'is', 'was',
        'were', 'are', 'product', 'item', 'one', 'get', 'use', 'like', 'really', 'got', 'buy', 'ordered'
    }
    
    # Tokenize words and count frequencies
    words = re.findall(r'\b\w+\b', text.lower())
    words = [w for w in words if w not in stopwords and len(w) > 2]
    
    word_freq = Counter(words)
    if not word_freq:
        # If no keywords found, just return the first few sentences
        return " ".join(sentences[:num_sentences])
        
    max_freq = max(word_freq.values())
    
    # Normalize frequencies
    for word in word_freq:
        word_freq[word] = word_freq[word] / max_freq
        
    # Score sentences
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_words = re.findall(r'\b\w+\b', sentence.lower())
        score = 0
        word_count = 0
        for word in sentence_words:
            if word in word_freq:
                score += word_freq[word]
                word_count += 1
        # Normalize by sentence length to avoid bias towards long sentences
        if word_count > 0:
            sentence_scores[i] = score / math.sqrt(word_count)
        else:
            sentence_scores[i] = 0
            
    # Sort sentences by score and pick top N
    top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    # Keep the original order of sentences
    top_indices.sort()
    
    summary = " ".join([sentences[idx] for idx in top_indices])
    return summary

def generate_summary(reviews_list, max_len=100):
    """
    Generates a concise summary from a list of reviews.
    Uses Hugging Face model as primary and TF-IDF Extractive summarization as fallback.
    Returns (summary_text, model_used).
    """
    cleaned_reviews = [str(r).strip() for r in reviews_list if str(r).strip()]
    
    if not cleaned_reviews:
        return "No reviews available to summarize.", "None"
        
    if len(cleaned_reviews) <= 2:
        return " ".join(cleaned_reviews), "Original Reviews (Too short to summarize)"
        
    # Join reviews into a single text block
    full_text = " ".join(cleaned_reviews)
    
    # Try using Hugging Face T5 Summarizer if enabled
    try:
        engine_choice = st.session_state.get("summarization_engine", "Fast NLP (Extractive)")
    except Exception:
        engine_choice = "Fast NLP (Extractive)"
        
    summarizer, hf_success = None, False
    if engine_choice == "Advanced AI (T5 Transformer)":
        summarizer, hf_success = load_hf_summarization_pipeline()
    
    if hf_success and summarizer is not None:
        try:
            # T5 requires structured inputs sometimes, or we can just pass text.
            # Truncate text to avoid model context length errors (e.g. 512 tokens)
            input_text = "summarize: " + full_text[:1024]
            # Set dynamic max/min length
            max_length = min(max_len, max(30, int(len(input_text.split()) * 0.5)))
            min_length = min(15, max_length - 5)
            
            res = summarizer(
                input_text, 
                max_length=max_length, 
                min_length=min_length, 
                do_sample=False
            )
            summary_text = res[0]['summary_text']
            
            # Clean up T5 padding / artifacts if any
            summary_text = summary_text.replace("summarize: ", "").strip()
            
            # Capitalize first letter if not capitalized
            if summary_text and not summary_text[0].isupper():
                summary_text = summary_text[0].upper() + summary_text[1:]
                
            return summary_text, "Advanced AI (Transformer)"
        except Exception as e:
            # Fail silently and let fallback handle it
            pass
            
    # Fallback: High-Quality Sentence Extractor
    # Adjust number of sentences based on target summary length
    num_sentences = 2 if max_len < 80 else 3
    summary_text = fallback_extractive_summary(cleaned_reviews, num_sentences=num_sentences)
    return summary_text, "Fast NLP Engine (Extractive)"

def get_sentiment_summaries(df):
    """
    Generates summaries for:
    - All reviews
    - Positive reviews
    - Negative reviews
    Returns a dict with summaries and their respective metadata.
    """
    results = {}
    
    # All Reviews
    all_reviews = df["Review"].tolist()
    summary_all, model_all = generate_summary(all_reviews, max_len=120)
    results["all"] = {"summary": summary_all, "model": model_all}
    
    # Positive Reviews
    pos_reviews = df[df["Sentiment"] == "Positive"]["Review"].tolist()
    if pos_reviews:
        summary_pos, model_pos = generate_summary(pos_reviews, max_len=80)
        results["positive"] = {"summary": summary_pos, "model": model_pos}
    else:
        results["positive"] = {"summary": "No positive reviews available to summarize.", "model": "None"}
        
    # Negative Reviews
    neg_reviews = df[df["Sentiment"] == "Negative"]["Review"].tolist()
    if neg_reviews:
        summary_neg, model_neg = generate_summary(neg_reviews, max_len=80)
        results["negative"] = {"summary": summary_neg, "model": model_neg}
    else:
        results["negative"] = {"summary": "No negative reviews available to summarize.", "model": "None"}
        
    return results

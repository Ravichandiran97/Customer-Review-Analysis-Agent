import re
from collections import Counter

# Set of standard stopwords
STOPWORDS = {
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
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'product', 'item', 'one', 'get',
    'use', 'like', 'really', 'got', 'buy', 'ordered', 'purchased', 'laptop', 'device', 'phone', 'seller', 'bought'
}

# Mapping of related words to unified product features
FEATURE_MAPPING = {
    'battery': 'Battery Life',
    'power': 'Battery Life',
    'charge': 'Battery Life',
    'charger': 'Battery Life',
    'charging': 'Battery Life',
    'packaging': 'Packaging',
    'package': 'Packaging',
    'box': 'Packaging',
    'wrapped': 'Packaging',
    'wrap': 'Packaging',
    'delivery': 'Delivery',
    'shipping': 'Delivery',
    'ship': 'Delivery',
    'delivered': 'Delivery',
    'shipped': 'Delivery',
    'arrived': 'Delivery',
    'quality': 'Product Quality',
    'build': 'Product Quality',
    'material': 'Product Quality',
    'materials': 'Product Quality',
    'feel': 'Product Quality',
    'feels': 'Product Quality',
    'price': 'Price & Value',
    'cost': 'Price & Value',
    'value': 'Price & Value',
    'money': 'Price & Value',
    'overpriced': 'Price & Value',
    'service': 'Customer Service',
    'support': 'Customer Service',
    'customer': 'Customer Service',
    'support': 'Customer Service',
    'screen': 'Screen & Display',
    'display': 'Screen & Display',
    'sound': 'Sound Quality',
    'audio': 'Sound Quality',
    'performance': 'Performance',
    'speed': 'Performance',
    'fast': 'Performance'
}

def extract_top_keywords(reviews_list, top_n=10):
    """
    Extracts the top N product features/keywords from the review texts.
    Uses aspect mapping to group related terms and falls back to general term counts.
    Returns list of tuples (keyword, frequency).
    """
    keyword_counts = Counter()
    
    # Process reviews
    for review in reviews_list:
        if not isinstance(review, str):
            continue
            
        # Standardize and clean text
        text = review.lower()
        words = re.findall(r'\b[a-z]{3,}\b', text) # extract words with length >= 3
        
        # Track aspects mentioned in this review (unique per review to prevent multiple counts from same sentence)
        mentioned_aspects = set()
        cleaned_words = []
        
        for w in words:
            if w in STOPWORDS:
                continue
                
            # If word maps to a feature, add the unified feature name
            if w in FEATURE_MAPPING:
                mentioned_aspects.add(FEATURE_MAPPING[w])
            else:
                cleaned_words.append(w)
                
        # Count unified features
        for aspect in mentioned_aspects:
            keyword_counts[aspect] += 1
            
        # Count regular words (if they aren't part of aspects)
        word_freq = Counter(cleaned_words)
        for w, f in word_freq.items():
            # Standardize casing for display (e.g. "Screen" instead of "screen")
            display_word = w.capitalize()
            keyword_counts[display_word] += f
            
    # Remove any generic stopwords that might have leaked, and return top_n
    filtered_counts = []
    for k, v in keyword_counts.most_common():
        if k.lower() in STOPWORDS:
            continue
        filtered_counts.append((k, v))
        if len(filtered_counts) == top_n:
            break
            
    # Default keywords if empty
    if not filtered_counts:
        return [
            ("Product Quality", 0),
            ("Battery Life", 0),
            ("Delivery", 0),
            ("Packaging", 0),
            ("Price & Value", 0)
        ]
        
    return filtered_counts

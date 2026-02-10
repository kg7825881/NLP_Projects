# analysis_engine.py
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline
import spacy

# Load models once to save time
nlp = spacy.load("en_core_web_sm")
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Feature 1: Emotion Detection (HuggingFace)
# Using a small, fast model for emotion detection
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

def analyze_sentiment_vader(text):
    """Basic Positive/Negative/Neutral score"""
    return sia.polarity_scores(text)['compound']

def detect_emotion(text):
    """New Feature: Detects Joy, Anger, Sadness, etc."""
    try:
        # Truncate text to 512 tokens for BERT models
        result = emotion_classifier(text[:512])[0] 
        return result['label']
    except:
        return "Neutral"

def extract_aspects(text):
    """New Feature: Aspect-Based Sentiment Analysis using Dependency Parsing"""
    doc = nlp(text.lower())
    aspects = []
    
    for token in doc:
        # Look for nouns (potential features)
        if token.pos_ == "NOUN":
            # Check if the noun has an adjective describing it
            for child in token.children:
                if child.pos_ == "ADJ":
                    aspects.append(f"{token.text} ({child.text})")
    
    return ", ".join(aspects) if aspects else "General"

def generate_smart_reply(review_text, sentiment_score):
    """New Feature: Drafts a customer service reply"""
    if sentiment_score >= 0.5:
        return f"Thank you for the glowing review! We are thrilled you enjoyed the {review_text[:20]}..."
    elif sentiment_score <= -0.5:
        return "We are incredibly sorry to hear about this experience. Please DM us your order ID so we can fix this immediately."
    else:
        return "Thank you for your feedback. We are constantly trying to improve and appreciate your input."
    
    # --- Add this to the bottom of analysis_engine.py ---

def load_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Smart Column Renaming
            # We look for common names and map them to 'Review'
            column_mapping = {
                'reviewText': 'Review',
                'text': 'Review',
                'content': 'Review',
                'summary': 'Review',
                'overall': 'Rating',
                'rating': 'Rating'
            }
            df.rename(columns=column_mapping, inplace=True)
            
            # Verify 'Review' exists
            if 'Review' not in df.columns:
                # Last resort: grab the column with the longest text
                text_col = df.select_dtypes(include=['object']).apply(lambda x: x.str.len().mean()).idxmax()
                df.rename(columns={text_col: 'Review'}, inplace=True)

            # Drop missing rows
            df.dropna(subset=['Review'], inplace=True)
            
            # Ensure it's a string
            df['Review'] = df['Review'].astype(str)
            
            return df
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return pd.DataFrame()
            
    else:
        # Demo data (Your existing fallback code)
        data = {
            'Review': [
                "The battery life is amazing but the camera sucks.",
                "Delivery was terrible! Arrived 3 days late.",
                "I absolutely love the design, it's so sleek.",
                "Waste of money. Stopped working after a week.",
                "Customer service was helpful when I called."
            ],
            'Rating': [3, 1, 5, 1, 4]
        }
        return pd.DataFrame(data)
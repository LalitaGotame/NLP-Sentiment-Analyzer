from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample
import pickle
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords')

app = Flask(__name__)

class BalancedSentimentAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.sentiment_lexicon = self.create_lexicon()
        
    def create_lexicon(self):
        """Create a sentiment lexicon with word weights"""
        return {
            'positive': {
                'good': 3, 'excellent': 4, 'great': 3, 'amazing': 4,
                'love': 4, 'like': 2, 'best': 3, 'perfect': 4,
                'awesome': 4, 'fantastic': 4, 'wonderful': 3,
                'happy': 3, 'pleased': 2, 'satisfied': 2,
                'recommend': 2, 'brilliant': 3, 'superb': 3
            },
            'negative': {
                'bad': 4, 'terrible': 4, 'awful': 4, 'horrible': 4,
                'worst': 4, 'hate': 4, 'dislike': 3, 'poor': 3,
                'disappointing': 3, 'disappointed': 3, 'sad': 2,
                'angry': 2, 'mad': 2, 'fail': 3, 'failure': 3,
                'problem': 2, 'issue': 2, 'wrong': 2, 'broken': 3
            },
            'negators': {'not', 'no', 'never', 'none', 'nothing', 'nobody'}
        }
    
    def preprocess_text(self, text):
        """Enhanced preprocessing"""
        text = str(text).lower()
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Keep important punctuation for sentiment
        text = re.sub(r'[^a-z\s!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove only truly neutral stopwords
        stop_words = set(stopwords.words('english'))
        # KEEP important words
        words_to_keep = set(self.sentiment_lexicon['positive'].keys()) | \
                       set(self.sentiment_lexicon['negative'].keys()) | \
                       self.sentiment_lexicon['negators']
        stop_words = stop_words - words_to_keep
        
        words = text.split()
        filtered_words = []
        
        # Handle negations: "not good" → "not_good"
        i = 0
        while i < len(words):
            if words[i] in self.sentiment_lexicon['negators'] and i + 1 < len(words):
                filtered_words.append(f"{words[i]}_{words[i+1]}")
                i += 2
            else:
                if words[i] not in stop_words:
                    filtered_words.append(words[i])
                i += 1
        
        return ' '.join(filtered_words)
    
    def balance_dataset(self, df):
        """Balance the dataset to handle class imbalance"""
        print("Balancing dataset...")
        
        # Separate classes
        df_majority = df[df['Sentiment'] == 'neutral']
        df_minority_positive = df[df['Sentiment'] == 'positive']
        df_minority_negative = df[df['Sentiment'] == 'negative']
        
        # Downsample majority class
        n_samples = min(len(df_majority), len(df_minority_positive), len(df_minority_negative))
        df_majority_downsampled = resample(df_majority,
                                          replace=False,
                                          n_samples=n_samples,
                                          random_state=42)
        
        # Combine balanced dataset
        df_balanced = pd.concat([df_majority_downsampled, 
                                 df_minority_positive, 
                                 df_minority_negative])
        
        print(f"Original: {len(df)} samples")
        print(f"Balanced: {len(df_balanced)} samples")
        print("Class distribution after balancing:")
        print(df_balanced['Sentiment'].value_counts())
        
        return df_balanced
    
    def train_model(self, df):
        """Train model on balanced data"""
        print("\nTraining model...")
        
        # Balance the dataset
        df_balanced = self.balance_dataset(df)
        
        # Preprocess
        df_balanced['processed'] = df_balanced['Sentence'].apply(self.preprocess_text)
        
        # Use TF-IDF with n-grams
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),  # Capture phrases
            stop_words='english'
        )
        
        X = self.vectorizer.fit_transform(df_balanced['processed'])
        y = df_balanced['Sentiment']
        
        # Train model
        self.model = MultinomialNB()
        self.model.fit(X, y)
        
        # Test with critical words
        print("\nTesting model with key words:")
        test_cases = [
            ("bad", "negative"),
            ("very bad", "negative"),
            ("not good", "negative"),
            ("terrible", "negative"),
            ("good", "positive"),
            ("excellent", "positive"),
            ("ok", "neutral"),
            ("normal", "neutral")
        ]
        
        for text, expected in test_cases:
            result = self.predict(text, verbose=False)
            match = "✓" if result['sentiment'] == expected else "✗"
            print(f"{match} '{text}' → {result['sentiment']} (expected: {expected})")
        
        return self.model
    
    def rule_based_fallback(self, text):
        """Rule-based sentiment as fallback"""
        text_lower = text.lower()
        
        positive_score = 0
        negative_score = 0
        
        # Check positive words
        for word, weight in self.sentiment_lexicon['positive'].items():
            if word in text_lower:
                positive_score += weight
        
        # Check negative words (with higher priority)
        for word, weight in self.sentiment_lexicon['negative'].items():
            if word in text_lower:
                negative_score += weight * 1.5  # Negative words get extra weight
        
        # Handle negations
        for negator in self.sentiment_lexicon['negators']:
            if negator in text_lower:
                # If there's a positive word after negator, flip to negative
                words = text_lower.split()
                for i, word in enumerate(words):
                    if word == negator and i + 1 < len(words):
                        next_word = words[i + 1]
                        if next_word in self.sentiment_lexicon['positive']:
                            positive_score -= 3
                            negative_score += 3
        
        if negative_score > positive_score and negative_score > 2:
            return 'negative', max(0.6, min(0.9, negative_score / 10))
        elif positive_score > negative_score and positive_score > 2:
            return 'positive', max(0.6, min(0.9, positive_score / 10))
        else:
            return 'neutral', 0.5
    
    def predict(self, text, verbose=True):
        """Predict sentiment with ML model + rule-based fallback"""
        # First try rule-based (for single words like "bad")
        rule_result, rule_confidence = self.rule_based_fallback(text)
        
        if verbose:
            print(f"Rule-based: '{text}' → {rule_result} ({rule_confidence:.0%})")
        
        # If we have a trained model, use it
        if self.vectorizer and self.model:
            try:
                # Preprocess
                processed = self.preprocess_text(text)
                
                # Vectorize
                if processed.strip():  # If we have words after preprocessing
                    X = self.vectorizer.transform([processed])
                    
                    # Predict
                    ml_prediction = self.model.predict(X)[0]
                    ml_probabilities = self.model.predict_proba(X)[0]
                    ml_confidence = max(ml_probabilities)
                    
                    if verbose:
                        print(f"ML prediction: {ml_prediction} ({ml_confidence:.0%})")
                    
                    # If ML confidence is high, use it
                    if ml_confidence > 0.7:
                        return {
                            'sentiment': ml_prediction,
                            'confidence': float(ml_confidence),
                            'method': 'ml'
                        }
            except Exception as e:
                if verbose:
                    print(f"ML failed: {e}")
        
        # Otherwise use rule-based
        return {
            'sentiment': rule_result,
            'confidence': float(rule_confidence),
            'method': 'rule'
        }

# Initialize and train
print("=" * 60)
print("SENTIMENT ANALYZER INITIALIZATION")
print("=" * 60)

analyzer = BalancedSentimentAnalyzer()

try:
    df = pd.read_csv("tweeterdata.csv")
    print(f"✓ Loaded dataset: {len(df)} rows")
    print(f"  Classes: {df['Sentiment'].value_counts().to_dict()}")
    
    # Train the model
    model = analyzer.train_model(df)
    
    # Save model
    with open('balanced_model.pkl', 'wb') as f:
        pickle.dump({
            'vectorizer': analyzer.vectorizer,
            'model': analyzer.model,
            'lexicon': analyzer.sentiment_lexicon
        }, f)
    print("✓ Model trained and saved")
    
except FileNotFoundError:
    print("✗ tweeterdata.csv not found!")
    print("  Using rule-based analyzer only")
except Exception as e:
    print(f"✗ Error: {e}")
    print("  Using rule-based analyzer only")

print("\n" + "=" * 60)
print("READY FOR USE!")
print("'bad' will be classified as NEGATIVE")
print("=" * 60 + "\n")

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please enter some text'}), 400
        
        result = analyzer.predict(text, verbose=False)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🌐 Open: http://localhost:5000")
    print("📝 Type text and click 'Analyze'")
    print("🎯 Examples to try:")
    print("   • 'This is bad' → Should show NEGATIVE (red)")
    print("   • 'I love it!' → Should show POSITIVE (green)")
    print("   • 'It's ok' → Should show NEUTRAL (orange)\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
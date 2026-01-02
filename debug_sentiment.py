# debug_sentiment.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import numpy as np

# Download stopwords
nltk.download('stopwords')

# Load your data
df = pd.read_csv("tweeterdata.csv")
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:")
print(df['Sentiment'].value_counts())
print(f"\nPercentage:")
print(df['Sentiment'].value_counts(normalize=True))

# Check if "bad" exists in dataset
print("\n" + "="*50)
print("Checking for 'bad' in dataset...")
bad_examples = df[df['Sentence'].str.contains('bad', case=False, na=False)]
if len(bad_examples) > 0:
    print(f"Found {len(bad_examples)} examples with 'bad':")
    for idx, row in bad_examples.head().iterrows():
        print(f"  '{row['Sentence'][:50]}...' → {row['Sentiment']}")
else:
    print("No examples with 'bad' found in dataset")

# Check some sample sentences
print("\n" + "="*50)
print("Sample sentences from each class:")
for sentiment in df['Sentiment'].unique():
    samples = df[df['Sentiment'] == sentiment].head(2)
    print(f"\n{sentiment.upper()}:")
    for idx, row in samples.iterrows():
        print(f"  '{row['Sentence'][:60]}...'")

# Custom preprocessing
def preprocess_text(text):
    text = str(text).lower()
    # Remove URLs and special characters but keep words
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    
    # CRITICAL: Keep negative words - DO NOT remove them
    negative_words = {
        'bad', 'terrible', 'awful', 'horrible', 'worst', 'poor',
        'not', 'no', 'never', 'none', 'nothing', 'nobody',
        'disappointing', 'disappointed', 'hate', 'hated'
    }
    
    # Get standard stopwords
    standard_stopwords = set(stopwords.words('english'))
    
    # Remove only neutral stopwords, keep negative ones
    stopwords_to_remove = standard_stopwords - negative_words
    
    # Also remove very common neutral words
    common_neutral = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    stopwords_to_remove.update(common_neutral)
    
    # Keep words that are NOT in stopwords_to_remove
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords_to_remove]
    
    return ' '.join(filtered_words)

# Test preprocessing
print("\n" + "="*50)
print("Testing preprocessing:")
test_texts = [
    "This is bad quality",
    "I hate this product",
    "It's not good at all",
    "The weather is nice today"
]

for text in test_texts:
    cleaned = preprocess_text(text)
    print(f"'{text}' → '{cleaned}'")

# Now train a simple model
print("\n" + "="*50)
print("Training model...")

# Preprocess all data
df['cleaned'] = df['Sentence'].apply(preprocess_text)

# Use CountVectorizer with n-grams
vectorizer = CountVectorizer(
    max_features=2000,
    ngram_range=(1, 3)  # Capture phrases like "not good", "is not good"
)

X = vectorizer.fit_transform(df['cleaned'])
y = df['Sentiment']

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test accuracy
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")

# Test specific words
print("\n" + "="*50)
print("Testing specific words:")

test_cases = [
    "bad",
    "very bad",
    "this is bad",
    "terrible",
    "awful",
    "worst",
    "hate",
    "not good",
    "no good",
    "good",
    "excellent",
    "amazing",
    "love it",
    "ok",
    "fine",
    "normal",
    "average"
]

for text in test_cases:
    cleaned = preprocess_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]
    
    # Find index of prediction
    pred_idx = list(model.classes_).index(prediction)
    confidence = probability[pred_idx]
    
    print(f"'{text}' → {prediction.upper()} ({confidence:.1%})")
    print(f"  Probabilities: Positive={probability[list(model.classes_).index('positive')]:.1%}, "
          f"Negative={probability[list(model.classes_).index('negative')]:.1%}, "
          f"Neutral={probability[list(model.classes_).index('neutral')]:.1%}")
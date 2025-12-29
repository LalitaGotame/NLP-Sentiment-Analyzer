
from flask import Flask, render_template, request 
import pickle 
import re 
import string 
from nltk.corpus import stopwords
import nltk 


app = Flask(__name__) 

# Load trained model and vectorizer
classifier = pickle.load(open("sentiment_model.pkl", "rb")) 
vectorizer = pickle.load(open("vectorizer.pkl", "rb")) 
stop_words = set(stopwords.words('english')) 

def preprocess_text(text): 
    text = text.lower() 
    text = re.sub(r'\d+', '', text) 
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text = " ".join([word for word in text.split() if word not in stop_words]) 
    return text 

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        user_text = request.form["text"]
        processed_text = preprocess_text(user_text)

        
        print("Processed text:", processed_text)

        vector = vectorizer.transform([processed_text])
        print("Vector sum:", vector.sum())
        prediction = classifier.predict(vector)[0]

        if prediction == 1:
            sentiment = "Positive 😊"
        else:
            sentiment = "Negative 😞"

    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
import numpy as np
import pandas as pd
import re
import string
import pickle


#load model
with open("static/model/trained_model.pickle","rb") as f:
    model = pickle.load(f)

# Load the vectorizer
with open("static/model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

#load stopwords
with open("static/model/corpora/stopwords/english","r") as file:
    sw = file.read().splitlines()

from nltk.stem import PorterStemmer
port_stem = PorterStemmer()


def stemming(content):
  content = " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in content.split())
  stemmed_content = re.sub('[^a-zA-Z]',' ',content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in sw]
  stemmed_content = ' '.join(stemmed_content)

  return stemmed_content

# Prediction pipeline function
def predict_sentiment(text):
    # Preprocess and stem the input text
    cleaned_text = stemming(text)
    
    # Vectorize the cleaned text
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Predict the sentiment using the trained model
    prediction = model.predict(vectorized_text)[0]
    
    # Return human-readable sentiment
    if prediction == 1:
        return "Positive"
    else:
        return "Negative"
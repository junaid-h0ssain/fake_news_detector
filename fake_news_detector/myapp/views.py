from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

import joblib
import os
import nltk
from nltk.corpus import stopwords

# Ensure NLTK data is downloaded 
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError: # type: ignore
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError: # type: ignore
    nltk.download('punkt')

model = os.path.join(os.path.dirname(__file__), 'models/fake_news_model.pkl')
vectoriser = os.path.join(os.path.dirname(__file__), 'models/tfidf_vectorizer.pkl')

# need to use joblib to use the model
try:
    loaded_model = joblib.load(model)
    loaded_vectorizer = joblib.load(vectoriser)
    print("ML model and vectorizer loaded successfully!")
except FileNotFoundError:
    print('Model not found in folder')
    loaded_model = None
    loaded_vectorizer = None

stop_words = set(stopwords.words('english'))

def prepro(text):
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    text = " ".join(tokens)
    return text

def predict(article_text):
    if loaded_model and loaded_vectorizer:
        cleaned_text = prepro(article_text)
        text_vector = loaded_vectorizer.transform([cleaned_text])
        # Make prediction
        prediction = loaded_model.predict(text_vector)[0]    
        probability = loaded_model.predict_proba(text_vector)[0]
        if prediction == 1:
            return "Fake News", probability[1]
        else:
            return "Real News", probability[0]
    else:
        return "Model not loaded. Cannot make prediction."

# myapp/views.py

def fake_news_detector(request):
    prediction_result = None
    confidence_score = None
    input_article = ''

    if request.method == 'POST':
        input_article = request.POST.get('article_text', '')
        if input_article:
            prediction, confidence = predict(input_article)
            prediction_result = prediction
            confidence_score = f"{confidence:.2f}" 
        else:
            prediction_result = "Please paste an article to analyze."

    return render(request, 'myapp/detector.html', {
        'prediction_result': prediction_result,
        'confidence_score': confidence_score,
        'input_article': input_article
    })
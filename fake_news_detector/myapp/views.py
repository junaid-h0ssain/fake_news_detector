from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

# myapp/utils.py or directly in myapp/views.py
import joblib
import os
import re
import string
import nltk
from nltk.corpus import stopwords

# Ensure NLTK data is downloaded (you might do this once during deployment setup)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')


# Load your trained model and vectorizer
# Adjust path as necessary for your Django project structure
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/fake_news_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'models/tfidf_vectorizer.pkl')

try:
    loaded_model = joblib.load(MODEL_PATH)
    loaded_vectorizer = joblib.load(VECTORIZER_PATH)
    print("ML model and vectorizer loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model or vectorizer file not found at {MODEL_PATH} or {VECTORIZER_PATH}. Make sure to train and save them first.")
    loaded_model = None
    loaded_vectorizer = None


stop_words = set(stopwords.words('english'))

def preprocess_text_for_prediction(text):
    # Apply the same preprocessing steps as during training
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def predict_fake_news(article_text):
    if loaded_model and loaded_vectorizer:
        cleaned_text = preprocess_text_for_prediction(article_text)
        # Vectorize the input text
        text_vectorized = loaded_vectorizer.transform([cleaned_text])
        # Make prediction
        prediction = loaded_model.predict(text_vectorized)[0]
        # Get probability (optional, but good for confidence score)
        probability = loaded_model.predict_proba(text_vectorized)[0]

        # Map prediction to human-readable label
        if prediction == 1:
            return "Fake News", probability[1]
        else:
            return "Real News", probability[0]
    else:
        return "Model not loaded. Cannot make prediction.", 0.0

# myapp/views.py
from django.shortcuts import render

def fake_news_detector(request):
    prediction_result = None
    confidence_score = None
    input_article = ""

    if request.method == 'POST':
        input_article = request.POST.get('article_text', '')
        if input_article:
            prediction, confidence = predict_fake_news(input_article)
            prediction_result = prediction
            confidence_score = f"{confidence:.2f}" # Format to 2 decimal places
        else:
            prediction_result = "Please paste an article to analyze."

    return render(request, 'myapp/detector.html', {
        'prediction_result': prediction_result,
        'confidence_score': confidence_score,
        'input_article': input_article
    })
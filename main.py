import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import re
import string

# Load datasets
true_news = pd.read_csv('data/True.csv')
fake_news = pd.read_csv('data/Fake.csv')

# Add labels
true_news['label'] = 0  # 0 for real news
fake_news['label'] = 1  # 1 for fake news

# Combine datasets
df = pd.concat([true_news, fake_news]).sample(frac=1, random_state=42).reset_index(drop=True)

# Drop unnecessary columns (date, subject might be useful for advanced features, but start simple)
df = df.drop(['subject', 'date'], axis=1)

# Fill any potential NaN values in 'text' or 'title'
df['text'] = df['text'].fillna('')
df['title'] = df['title'].fillna('')

# Combine title and text
df['full_text'] = df['title'] + " " + df['text']

print(df.head())
print(df['label'].value_counts())

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower() # Lowercasing
    text = re.sub(r'\[.*?\]', '', text) # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>+', '', text) # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub(r'\n', '', text) # Remove newline characters
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers

    # Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    text = " ".join(tokens)
    return text

df['cleaned_text'] = df['full_text'].apply(preprocess_text)
print(df[['full_text', 'cleaned_text']].head())

vectorizer = TfidfVectorizer(max_features=5000) # Limit features for simplicity
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
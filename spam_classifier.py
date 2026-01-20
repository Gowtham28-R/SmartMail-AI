# =========================
# Email Spam Classifier
# =========================

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------
# Download NLTK resources (run once)
# -------------------------
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')
    nltk.download('punkt')

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("dataset/spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

print("\nDataset loaded successfully!")
print(df.head())
print("\nClass distribution:\n", df['label'].value_counts())

# -------------------------
# Text Preprocessing
# -------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

# -------------------------
# Feature Extraction
# -------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Train Models
# -------------------------
nb = MultinomialNB()
lr = LogisticRegression(max_iter=1000)
svm = LinearSVC()

nb.fit(X_train, y_train)
lr.fit(X_train, y_train)
svm.fit(X_train, y_train)

# -------------------------
# Evaluate Models
# -------------------------
models = {
    "Naive Bayes": nb,
    "Logistic Regression": lr,
    "SVM": svm
}

print("\n===== MODEL RESULTS =====")
for name, model in models.items():
    y_pred = model.predict(X_test)
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# -------------------------
# Confusion Matrix (SVM)
# -------------------------
y_pred = svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM")
plt.show()

# -------------------------
# Save Best Model
# -------------------------
joblib.dump(svm, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nModel and vectorizer saved!")

# -------------------------
# Real-Time Prediction System
# -------------------------
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_email(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    result = model.predict(vec)[0]
    return "SPAM 🚫" if result == 1 else "HAM ✅"

print("\n===== REAL-TIME EMAIL CLASSIFIER =====")

while True:
    msg = input("\nEnter email text (or type 'exit'): ")
    if msg.lower() == "exit":
        print("Exiting program...")
        break
    print("Prediction:", predict_email(msg))

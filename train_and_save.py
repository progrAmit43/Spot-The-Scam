import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load the training data
url = 'https://coding-platform.s3.amazonaws.com/dev/lms/tickets/4c8465f0-fce0-484f-8497-d25feaa8e995/NqndMEyZakuimmFI.csv'
df = pd.read_csv(url)

# Preprocessing
df.dropna(subset=['description', 'fraudulent'], inplace=True)
X = df['description']
y = df['fraudulent']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer + Logistic Regression
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("✅ Model and vectorizer saved as 'model.joblib' and 'vectorizer.joblib'")

# ====== Fake Job Detection Ready-to-Run ======

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Load dataset
# Make sure this file path correct-a irukku
df = pd.read_csv(r"/content/fake_job_postings.csv", engine='python') # Added engine='python' to handle parsing issues

# Fill NaN values in 'description' with empty strings AND ensure string type
df['description'] = df['description'].fillna('').astype(str)

# 2️⃣ Features and labels
X = df['description']      # Job description text
y = df['fraudulent']       # 0 = Real, 1 = Fake

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5️⃣ Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6️⃣ Evaluate model
y_pred = model.predict(X_test_tfidf)
print("===== Model Accuracy =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred))

# 7️⃣ Test with a new job description
new_job = ["Looking for engineer, high pay, no interview required"]
new_job_tfidf = vectorizer.transform(new_job)
prediction = model.predict(new_job_tfidf)
print("\n===== New Job Prediction =====")
print("Prediction (0=Real, 1=Fake):", prediction[0])

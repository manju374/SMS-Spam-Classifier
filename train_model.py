import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load the updated dataset
df = pd.read_csv('spam_updated.csv')

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Improved TF-IDF + Naive Bayes pipeline
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2, ngram_range=(1, 2))),
    ('nb', MultinomialNB())
])

# Train model
model_pipeline.fit(X_train, y_train)

# Evaluate accuracy
accuracy = model_pipeline.score(X_test, y_test)
print(f"Model trained. Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model_pipeline, 'spam_classifier_model.pkl')
print("Model saved as 'spam_classifier_model.pkl'")

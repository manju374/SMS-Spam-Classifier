import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv('spam_updated.csv')

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2, ngram_range=(1, 2))),
    ('nb', MultinomialNB())
])

model_pipeline.fit(X_train, y_train)

accuracy = model_pipeline.score(X_test, y_test)
print(f"Model trained. Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model_pipeline, 'spam_classifier_model.pkl')
print("Model saved as 'spam_classifier_model.pkl'")

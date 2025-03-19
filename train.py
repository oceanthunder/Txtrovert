import pandas as pd
import json
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def load_doccano_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)

df = load_doccano_jsonl("admin.jsonl")

df.rename(columns={"text": "review", "label": "sentiment"}, inplace=True)

label_map = {"positive": 1, "negative": 0}
df["sentiment"] = df["sentiment"].apply(lambda x: label_map[x[0]] if isinstance(x, list) else label_map[x])

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

joblib.dump(model, "txtrovertModel.pkl")
joblib.dump(vectorizer, "tfidfVectorizer.pkl")

print("Model and vectorizer saved.")

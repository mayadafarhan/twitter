"""
Train Twitter Sentiment Analysis model
Faithful to original notebook preprocessing, but uses TF-IDF + Logistic Regression
for fast, reliable live demo (original used Keras LSTM).
"""
import re
import unicodedata
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Try NLTK for faithful preprocessing, fallback gracefully
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
    LEMM = WordNetLemmatizer()
    HAS_NLTK = True
    print("NLTK ready")
except Exception as e:
    print(f"NLTK unavailable ({e}), using fallback preprocessing")
    HAS_NLTK = False
    STOP_WORDS = set()
    LEMM = None


def process_text(text):
    """Same steps as original notebook cell 11"""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode('ascii', 'ignore').decode("utf-8", "ignore")
    text = text.lower()
    text = re.sub(r"https?//\S+|www\.\S+|\b\w+\.(com|ly|net|org|eg)\S*|\b\w+\.\w+\.(com|ly|net|org|eg)\S*", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z'\s]", ' ', text)
    text = re.sub(r"\s+", " ", text)

    if HAS_NLTK:
        try:
            words = word_tokenize(text)
            words = [LEMM.lemmatize(w) for w in words if w not in STOP_WORDS]
            words = [w for w in words if len(w) > 2]
            # unique preserving order (as in notebook)
            seen = set()
            final = []
            for w in words:
                if w not in seen:
                    seen.add(w)
                    final.append(w)
            return " ".join(final)
        except Exception:
            pass

    # Fallback: simple split
    words = [w for w in text.split() if len(w) > 2]
    return " ".join(words)


def main():
    print("Loading data...")
    df = pd.read_csv("twitter.csv")
    print(f"Shape: {df.shape}")
    print(df['labels'].value_counts())

    df = df.dropna().drop_duplicates()
    print(f"After cleaning: {df.shape}")

    print("Preprocessing text (this may take a minute)...")
    df['clean_text'] = df['text'].apply(process_text)
    df = df[df['clean_text'].str.strip() != ""]
    print(f"After empty-text removal: {df.shape}")

    X = df['clean_text']
    y = df['labels']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print("Building pipeline: TF-IDF (20k) + Logistic Regression...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=1000, C=5, multi_class='multinomial', solver='lbfgs', n_jobs=-1))
    ])

    print("Training...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=['Negative', 'Positive', 'Neutral', 'Irrelevant']).tolist()

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, "model.pkl")
    print("Saved model.pkl")

    metrics = {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "labels": ['Negative', 'Positive', 'Neutral', 'Irrelevant'],
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "total_size": int(len(df)),
        "vocab_size": len(pipeline.named_steps['tfidf'].vocabulary_),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics.json")

    # Save 200 sample tweets for batch demo (balanced)
    samples = df.sample(n=200, random_state=42)[['labels', 'text']]
    samples.to_csv("samples.csv", index=False)
    print("Saved samples.csv")

    # Save dataset stats
    stats = {
        "label_counts": df['labels'].value_counts().to_dict(),
        "avg_text_length": float(df['text'].str.len().mean()),
        "avg_word_count": float(df['text'].str.split().str.len().mean()),
    }
    with open("dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    main()

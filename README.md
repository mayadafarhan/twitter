# 🐦 Twitter Sentiment Analysis — Live Demo

A live, interactive demo of the [Twitter Sentiment Analysis](https://github.com/mayadafarhan/twitter) ML/NLP project.

## Run it

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download(['punkt','punkt_tab','stopwords','wordnet','omw-1.4'])"
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then open http://localhost:8501

## Features

- 🏠 **Dashboard** — dataset stats & sentiment distribution
- 🔮 **Live Prediction** — analyze any tweet with step-by-step preprocessing visualization, confidence bars, and top contributing words
- 📊 **Batch Analysis** — analyze 200 sample tweets or upload your own CSV
- 📈 **Model Performance** — accuracy, per-class metrics, confusion matrix
- ⚙️ **How It Works** — full pipeline explanation

## Model

- **Dataset:** 75,680 tweets (Positive / Negative / Neutral / Irrelevant)
- **Preprocessing:** identical to the original notebook (Unicode normalize, URL/mention removal, tokenize, stop-word removal, lemmatization)
- **Model:** TF-IDF (20k vocab, 1–2 grams) + Multinomial Logistic Regression
- **Accuracy:** 83% on 13,894 held-out tweets

> The original notebook used a Keras LSTM model. This demo uses TF-IDF + Logistic Regression so predictions load and run instantly in the browser, while keeping the exact same preprocessing pipeline and dataset.

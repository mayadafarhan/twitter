"""
Twitter Sentiment Analysis — Live Demo
Original project: https://github.com/mayadafarhan/twitter
"""
import re
import unicodedata
import json
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- NLTK (same as training) ----------
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    STOP_WORDS = set(stopwords.words("english"))
    LEMM = WordNetLemmatizer()
    HAS_NLTK = True
except Exception:
    HAS_NLTK = False
    STOP_WORDS = set()
    LEMM = None


def process_text(text):
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
            seen = set(); final = []
            for w in words:
                if w not in seen:
                    seen.add(w); final.append(w)
            return " ".join(final)
        except Exception:
            pass
    return " ".join([w for w in text.split() if len(w) > 2])


# ---------- Load model ----------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_metrics():
    with open("metrics.json") as f:
        return json.load(f)

@st.cache_data
def load_stats():
    with open("dataset_stats.json") as f:
        return json.load(f)

model = load_model()
metrics = load_metrics()
stats = load_stats()

LABEL_STYLE = {
    "Positive":   {"emoji": "😊", "color": "#1DA1F2", "bg": "#e8f5fe", "desc": "Happy, optimistic, or favorable sentiment"},
    "Negative":   {"emoji": "😠", "color": "#e0245e", "bg": "#fdeef4", "desc": "Angry, sad, or unfavorable sentiment"},
    "Neutral":    {"emoji": "😐", "color": "#657786", "bg": "#f0f3f5", "desc": "Factual or emotionally neutral content"},
    "Irrelevant": {"emoji": "🤷", "color": "#f45d22", "bg": "#fff3ec", "desc": "Off-topic, spam, or not related to the subject"},
}

# ---------- Page config & styling ----------
st.set_page_config(page_title="Twitter Sentiment Analysis — Live Demo", page_icon="🐦", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1DA1F2 0%, #0d8bd9 50%, #14171A 100%);
        padding: 2rem 2.5rem; border-radius: 16px; color: white; margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; font-size: 2.2rem; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.85); margin: 0.4rem 0 0 0; font-size: 1.05rem; }
    .card { background: #ffffff; border: 1px solid #e1e8ed; border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; }
    .pred-box { border-radius: 16px; padding: 1.6rem; text-align: center; margin: 1rem 0; }
    .pred-box .emoji { font-size: 3.2rem; }
    .pred-box .label { font-size: 1.9rem; font-weight: 800; margin: 0.3rem 0; }
    .pred-box .conf { font-size: 1.1rem; opacity: 0.85; }
    .step-box { background: #f7f9fa; border-left: 4px solid #1DA1F2; border-radius: 8px; padding: 0.8rem 1rem; margin: 0.5rem 0; font-size: 0.95rem; }
    .token-chip { display: inline-block; background: #e8f5fe; color: #0d8bd9; border-radius: 20px; padding: 0.25rem 0.8rem; margin: 0.15rem; font-size: 0.85rem; font-weight: 600; }
    .word-pos { display: inline-block; background: #e8f5fe; color: #0d8bd9; border-radius: 8px; padding: 0.2rem 0.6rem; margin: 0.15rem; font-size: 0.85rem; }
    .word-neg { display: inline-block; background: #fdeef4; color: #e0245e; border-radius: 8px; padding: 0.2rem 0.6rem; margin: 0.15rem; font-size: 0.85rem; }
    .footer { text-align: center; color: #657786; font-size: 0.85rem; padding: 1.5rem 0 0.5rem 0; }
    div[data-testid="stMetricValue"] { font-size: 1.7rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg", width=60)
    st.title("🐦 Sentiment AI")
    st.caption("Twitter Sentiment Analysis\nLive Demo")
    st.divider()
    page = st.radio("Navigation", ["🏠 Dashboard", "🔮 Live Prediction", "📊 Batch Analysis", "📈 Model Performance", "⚙️ How It Works"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Model Info**")
    st.caption(f"Algorithm: TF-IDF + Logistic Regression\nVocabulary: {metrics['vocab_size']:,} words\nAccuracy: {metrics['accuracy']*100:.1f}%\nDataset: {metrics['total_size']:,} tweets")
    st.divider()
    st.caption("Original project:\ngithub.com/mayadafarhan/twitter")

# ---------- Header ----------
st.markdown("""
<div class="main-header">
    <h1>🐦 Twitter Sentiment Analysis</h1>
    <p>Machine Learning &amp; NLP system that classifies tweets as <b>Positive</b>, <b>Negative</b>, <b>Neutral</b>, or <b>Irrelevant</b> — trained on 75,680 real tweets.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PAGE 1: DASHBOARD
# ============================================================
if page == "🏠 Dashboard":
    st.subheader("📊 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tweets", f"{metrics['total_size']:,}")
    c2.metric("Vocabulary Size", f"{metrics['vocab_size']:,}")
    c3.metric("Model Accuracy", f"{metrics['accuracy']*100:.1f}%")
    c4.metric("Classes", "4")

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Sentiment Distribution**")
        dist = pd.Series(stats['label_counts'])
        fig = px.pie(values=dist.values, names=dist.index, color=dist.index,
                     color_discrete_map={"Positive": "#1DA1F2", "Negative": "#e0245e", "Neutral": "#657786", "Irrelevant": "#f45d22"},
                     hole=0.45, title="Tweets per sentiment class")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Class Breakdown**")
        for label, count in stats['label_counts'].items():
            s = LABEL_STYLE[label]
            pct = count / metrics['total_size'] * 100
            st.markdown(f"{s['emoji']} **{label}** — {count:,} tweets ({pct:.1f}%)")
            st.progress(count / max(stats['label_counts'].values()))
        st.divider()
        st.markdown(f"📏 Avg. tweet length: **{stats['avg_text_length']:.0f} characters**")
        st.markdown(f"🔤 Avg. words per tweet: **{stats['avg_word_count']:.1f}**")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**🎯 What this demo does**")
    st.markdown("""
    This live demo showcases a complete NLP pipeline — from raw tweet text to sentiment prediction:

    1. **Preprocessing** — Unicode normalization, URL/mention/hashtag removal, lowercasing, stop-word removal & lemmatization (same steps as the original notebook)
    2. **Feature Extraction** — TF-IDF vectorization with 20,000-word vocabulary and bigrams
    3. **Classification** — Multinomial Logistic Regression trained on 55,575 tweets
    4. **Evaluation** — Tested on 13,894 held-out tweets with **83% accuracy**

    👉 Go to **🔮 Live Prediction** to try it yourself, or **📊 Batch Analysis** to analyze hundreds of tweets at once.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 2: LIVE PREDICTION
# ============================================================
elif page == "🔮 Live Prediction":
    st.subheader("🔮 Analyze a Tweet in Real Time")
    st.caption("Type any tweet below and watch the full NLP pipeline work step by step.")

    EXAMPLES = {
        "😊 Positive": "I absolutely love this game! Best experience ever, so much fun and amazing graphics! 🎮",
        "😠 Negative": "This is terrible, worst update ever. I hate it, so disappointing and buggy. Waste of money 😡",
        "😐 Neutral": "The new update for the game was released today at 5 PM according to the official announcement.",
        "🤷 Irrelevant": "Rock-Hard La Varlope, RARE & POWERFUL, HANDSOME JACKPOT, Borderlands 3 (Xbox) dlvr.it/RMTrgF",
        "🔥 Borderlands": "that was the first borderlands session in a long time where i actually had a really satisfying combat experience. i got some really good kills",
    }

    selected_example = st.selectbox("Or try an example:", ["✏️ Write my own"] + list(EXAMPLES.keys()))
    default_text = EXAMPLES.get(selected_example, "") if selected_example != "✏️ Write my own" else ""
    tweet = st.text_area("✍️ Your tweet:", value=default_text, height=110, placeholder="Type or paste a tweet here...")

    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if not tweet.strip():
            st.warning("Please enter a tweet first.")
        else:
            with st.spinner("Running NLP pipeline..."):
                cleaned = process_text(tweet)
                tokens = cleaned.split() if cleaned else []
                proba = model.predict_proba([cleaned])[0]
                classes = list(model.classes_)
                pred_idx = int(np.argmax(proba))
                pred_label = classes[pred_idx]
                pred_conf = float(proba[pred_idx])
                s = LABEL_STYLE[pred_label]

                # --- Result ---
                st.markdown(f"""
                <div class="pred-box" style="background: {s['bg']}; border: 2px solid {s['color']};">
                    <div class="emoji">{s['emoji']}</div>
                    <div class="label" style="color: {s['color']};">{pred_label}</div>
                    <div class="conf">Confidence: <b>{pred_conf*100:.1f}%</b> — {s['desc']}</div>
                </div>
                """, unsafe_allow_html=True)

                # --- Confidence bars ---
                st.markdown("**📊 Confidence per class**")
                conf_df = pd.DataFrame({"Class": classes, "Confidence": proba * 100})
                conf_df["Class"] = conf_df["Class"].apply(lambda x: f"{LABEL_STYLE[x]['emoji']} {x}")
                fig = px.bar(conf_df, x="Confidence", y="Class", orientation="h",
                             color="Confidence", color_continuous_scale=["#fdeef4", "#1DA1F2"],
                             text=conf_df["Confidence"].apply(lambda x: f"{x:.1f}%"))
                fig.update_layout(showlegend=False, xaxis_title="Confidence (%)", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

                # --- Preprocessing steps ---
                st.markdown("**⚙️ Preprocessing Pipeline**")
                st.markdown(f'<div class="step-box"><b>1. Original:</b> {tweet[:220]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="step-box"><b>2. Cleaned (URLs, mentions, numbers, punctuation removed):</b> {cleaned if cleaned else "(empty)"}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="step-box"><b>3. Tokens after stop-word removal &amp; lemmatization ({len(tokens)} words):</b><br>' +
                            "".join([f'<span class="token-chip">{t}</span>' for t in tokens]) + '</div>', unsafe_allow_html=True)

                # --- Top contributing words ---
                st.markdown("**🔑 Words that influenced this prediction**")
                try:
                    tfidf = model.named_steps['tfidf']
                    clf = model.named_steps['clf']
                    feat_names = np.array(tfidf.get_feature_names_out())
                    coef = clf.coef_[pred_idx]
                    top_pos_idx = np.argsort(coef)[-8:][::-1]
                    top_neg_idx = np.argsort(coef)[:8]
                    st.markdown("Words pushing **toward** this class:")
                    st.markdown(" ".join([f'<span class="word-pos">{feat_names[i]}</span>' for i in top_pos_idx]), unsafe_allow_html=True)
                    st.markdown("Words pushing **away** from this class:")
                    st.markdown(" ".join([f'<span class="word-neg">{feat_names[i]}</span>' for i in top_neg_idx]), unsafe_allow_html=True)
                except Exception:
                    st.caption("Top-words explanation unavailable.")

# ============================================================
# PAGE 3: BATCH ANALYSIS
# ============================================================
elif page == "📊 Batch Analysis":
    st.subheader("📊 Batch Analysis")
    st.caption("Analyze hundreds of tweets at once — either the built-in sample set or your own CSV file.")

    tab1, tab2 = st.tabs(["📦 Sample Dataset (200 tweets)", "📁 Upload Your Own CSV"])
    results_df = None

    with tab1:
        if st.button("▶️ Analyze 200 Sample Tweets", type="primary"):
            samples = pd.read_csv("samples.csv")
            with st.spinner("Analyzing 200 tweets..."):
                cleaned_all = samples['text'].apply(process_text)
                preds = model.predict(cleaned_all)
                probas = model.predict_proba(cleaned_all)
                confs = probas.max(axis=1)
                samples = samples.copy()
                samples['predicted'] = preds
                samples['confidence'] = confs
                samples['correct'] = samples['labels'] == samples['predicted']
                results_df = samples
                st.session_state['batch_results'] = samples

    with tab2:
        uploaded = st.file_uploader("Upload a CSV with a 'text' column", type=["csv"])
        if uploaded is not None:
            try:
                up_df = pd.read_csv(uploaded)
                if 'text' not in up_df.columns:
                    st.error("CSV must contain a 'text' column.")
                else:
                    if st.button("▶️ Analyze Uploaded Tweets", type="primary"):
                        with st.spinner(f"Analyzing {len(up_df):,} tweets..."):
                            cleaned_all = up_df['text'].astype(str).apply(process_text)
                            preds = model.predict(cleaned_all)
                            probas = model.predict_proba(cleaned_all)
                            up_df = up_df.copy()
                            up_df['predicted'] = preds
                            up_df['confidence'] = probas.max(axis=1)
                            st.session_state['batch_results'] = up_df
                            results_df = up_df
            except Exception as e:
                st.error(f"Could not read file: {e}")

    if 'batch_results' in st.session_state:
        res = st.session_state['batch_results']
        st.divider()
        st.subheader("📈 Results")

        # Distribution chart
        dist = res['predicted'].value_counts()
        fig = px.pie(values=dist.values, names=dist.index,
                     color=dist.index,
                     color_discrete_map={"Positive": "#1DA1F2", "Negative": "#e0245e", "Neutral": "#657786", "Irrelevant": "#f45d22"},
                     hole=0.45, title="Predicted sentiment distribution")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Tweets Analyzed", f"{len(res):,}")
        c2.metric("Avg. Confidence", f"{res['confidence'].mean()*100:.1f}%")
        if 'correct' in res.columns:
            c3.metric("Accuracy on Samples", f"{res['correct'].mean()*100:.1f}%")
        else:
            c3.metric("Most Common", dist.index[0])

        st.markdown("**📋 Detailed Results**")
        display = res.copy()
        if 'correct' in display.columns:
            display = display[['text', 'labels', 'predicted', 'confidence', 'correct']]
            display.columns = ['Tweet', 'True Label', 'Predicted', 'Confidence', 'Correct']
        else:
            display = display[['text', 'predicted', 'confidence']]
            display.columns = ['Tweet', 'Predicted', 'Confidence']
        display['Confidence'] = (display['Confidence'] * 100).round(1)
        st.dataframe(display, use_container_width=True, hide_index=True)

        # Download
        csv_data = res.to_csv(index=False)
        st.download_button("⬇️ Download Results (CSV)", csv_data, "sentiment_results.csv", "text/csv")

# ============================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================
elif page == "📈 Model Performance":
    st.subheader("📈 Model Performance Evaluation")
    st.caption(f"Evaluated on {metrics['test_size']:,} held-out tweets (20% test split, stratified, random_state=42).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    c2.metric("Macro F1", f"{metrics['report']['macro avg']['f1-score']:.3f}")
    c3.metric("Weighted F1", f"{metrics['report']['weighted avg']['f1-score']:.3f}")
    c4.metric("Test Size", f"{metrics['test_size']:,}")

    # Per-class metrics
    st.markdown("**📋 Per-Class Metrics**")
    rows = []
    for label in metrics['labels']:
        r = metrics['report'][label]
        rows.append({"Class": f"{LABEL_STYLE[label]['emoji']} {label}", "Precision": round(r['precision'], 3),
                     "Recall": round(r['recall'], 3), "F1-Score": round(r['f1-score'], 3), "Support": int(r['support'])})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # F1 bar chart
    fig = px.bar(pd.DataFrame(rows), x="Class", y="F1-Score", color="F1-Score",
                 color_continuous_scale=["#e0245e", "#f45d22", "#657786", "#1DA1F2"],
                 title="F1-Score per class", text="F1-Score")
    fig.update_layout(showlegend=False, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix
    st.markdown("**🔲 Confusion Matrix**")
    cm = np.array(metrics['confusion_matrix'])
    fig_cm, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=metrics['labels'], yticklabels=metrics['labels'], ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix — Twitter Sentiment Classifier")
    st.pyplot(fig_cm)

# ============================================================
# PAGE 5: HOW IT WORKS
# ============================================================
elif page == "⚙️ How It Works":
    st.subheader("⚙️ How It Works")
    st.markdown("""
    ### 🔄 The NLP Pipeline

    ```
    Raw Tweet → Unicode Normalize → Lowercase → Remove URLs/@mentions/#hashtags/numbers
              → Tokenize → Remove Stop Words → Lemmatize → TF-IDF Vectorize → Logistic Regression → Sentiment
    ```

    ### 🧹 Step 1 — Text Preprocessing (from the original notebook)
    - **Unicode normalization** (NFKD) and ASCII conversion
    - **URL removal** (`https://...`, `www....`, `*.com/ly/net/org/eg`)
    - **Mention removal** (`@username`) and **hashtag unwrapping** (`#love` → `love`)
    - **Number & punctuation removal**, extra-whitespace cleanup
    - **Tokenization** with NLTK `word_tokenize`
    - **Stop-word removal** (English stop words like "the", "is", "a")
    - **Lemmatization** with WordNet (`loving` → `love`, `kills` → `kill`)
    - Keep words longer than 2 characters, deduplicate

    ### 🔢 Step 2 — Feature Extraction
    - **TF-IDF Vectorizer** with a 20,000-word vocabulary
    - **N-grams (1, 2)** — captures phrases like "not good" or "so much"
    - **Sublinear TF** — dampens very frequent words

    ### 🤖 Step 3 — Classification
    - **Multinomial Logistic Regression** (C=5, lbfgs solver)
    - Trained on **55,575 tweets**, tested on **13,894 tweets**
    - Outputs a **probability for each of the 4 classes** — the highest wins

    ### 📊 Results
    - **Accuracy: 83%** — strong performance for noisy social-media text
    - Balanced performance across all 4 classes (F1 between 0.78 and 0.86)

    ### 🔁 Original vs Live Demo
    | Aspect | Original Notebook | This Live Demo |
    |---|---|---|
    | Model | Keras LSTM (Embedding 100 → LSTM 100 → Dense) | TF-IDF + Logistic Regression |
    | Why the change? | Research/training | Fast loading & instant predictions in the browser |
    | Preprocessing | NLTK pipeline | **Identical** NLTK pipeline |
    | Dataset | twitter.csv (75,680 tweets) | **Same dataset** |
    """)

st.markdown('<div class="footer">🐦 Twitter Sentiment Analysis — Live Demo &nbsp;|&nbsp; Original project: github.com/mayadafarhan/twitter &nbsp;|&nbsp; Built with Python, Scikit-Learn, NLTK &amp; Streamlit</div>', unsafe_allow_html=True)

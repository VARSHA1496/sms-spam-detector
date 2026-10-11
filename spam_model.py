"""
spam_model.py
-------------
Small helper module used by the web app: loading the saved model and
turning a message into a prediction + an explanation.
"""
import json
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "spam_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def model_exists():
    return MODEL_PATH.exists() and METRICS_PATH.exists()


def load_model():
    """Load the trained scikit-learn Pipeline (TF-IDF  ->  Naive Bayes)."""
    return joblib.load(MODEL_PATH)


def load_metrics():
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


def predict_message(pipeline, text):
    """Return ('spam' or 'ham', probability that the message is spam)."""
    probabilities = pipeline.predict_proba([text])[0]
    classes = list(pipeline.classes_)
    spam_probability = float(probabilities[classes.index("spam")])
    label = "spam" if spam_probability >= 0.5 else "ham"
    return label, spam_probability


def explain_message(pipeline, text, top_n=8):
    """
    Find which words of the message pushed the model towards 'spam' or 'ham'.

    For every word w that the model knows, its influence is
        tfidf(w in message) * ( log P(w | spam) - log P(w | ham) )
    Positive = looks like spam, negative = looks like normal text.
    """
    vectorizer = pipeline.named_steps["tfidf"]
    nb = pipeline.named_steps["nb"]
    classes = list(nb.classes_)
    log_ratio = (
        nb.feature_log_prob_[classes.index("spam")]
        - nb.feature_log_prob_[classes.index("ham")]
    )
    names = vectorizer.get_feature_names_out()
    row = vectorizer.transform([text]).tocoo()

    spam_words, ham_words = [], []
    for column, value in zip(row.col, row.data):
        word = names[column]
        if " " in word:               # skip two-word features, keep single words
            continue
        score = float(value * log_ratio[column])
        if score > 0:
            spam_words.append((word, score))
        elif score < 0:
            ham_words.append((word, score))

    spam_words.sort(key=lambda item: -item[1])
    ham_words.sort(key=lambda item: item[1])
    return spam_words[:top_n], ham_words[:top_n]

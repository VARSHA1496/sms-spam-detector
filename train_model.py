"""
train_model.py
--------------
Trains the SMS spam detector and saves it to the  model/  folder.

    python train_model.py

Steps:  load data -> split into train/test -> TF-IDF -> Multinomial Naive Bayes
        -> evaluate on the unseen test set -> save model + metrics.
"""
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from data_loader import load_dataset
from spam_model import METRICS_PATH, MODEL_DIR, MODEL_PATH

RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",     # ignore words like "the", "and", "is"
            ngram_range=(1, 2),       # single words and word pairs
            min_df=2,                 # ignore words seen in only 1 message
            sublinear_tf=True,
        )),
        ("nb", MultinomialNB(alpha=0.1)),
    ])


def top_words(pipeline, X_train, y_train, label, n=10):
    """Words with the highest average TF-IDF weight in messages of one class."""
    vectorizer = pipeline.named_steps["tfidf"]
    names = vectorizer.get_feature_names_out()
    X = vectorizer.transform(X_train[y_train == label])
    means = X.mean(axis=0).A1
    ranked = sorted(zip(names, means), key=lambda item: -item[1])
    singles = [(w, float(s)) for w, s in ranked if " " not in w]
    return singles[:n]


def train_and_save(verbose=True):
    log = print if verbose else (lambda *args, **kwargs: None)

    log("Loading dataset ...")
    df = load_dataset()
    total_raw = len(df)
    df = df.drop_duplicates(subset="message").reset_index(drop=True)
    log(f"  {total_raw} messages, {total_raw - len(df)} exact duplicates removed "
        f"-> {len(df)} unique messages")
    log(f"  not spam (ham): {(df.label == 'ham').sum()}   spam: {(df.label == 'spam').sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"],
    )
    log(f"  training messages: {len(X_train)}   test messages: {len(X_test)}")

    pipeline = build_pipeline()

    log("Cross-validating on the training set (5 folds) ...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")

    log("Training the final model ...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label="spam", zero_division=0)
    recall = recall_score(y_test, predictions, pos_label="spam", zero_division=0)
    f1 = f1_score(y_test, predictions, pos_label="spam", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=["ham", "spam"]).ravel()

    log("\n=== RESULTS ON THE UNSEEN TEST SET ===")
    log(f"Accuracy : {accuracy * 100:.2f}%")
    log(f"Precision: {precision * 100:.2f}%   (of messages flagged spam, how many really were)")
    log(f"Recall   : {recall * 100:.2f}%   (of all real spam, how much was caught)")
    log(f"F1-score : {f1 * 100:.2f}%")
    log(f"5-fold cross-validation accuracy: {cv_scores.mean() * 100:.2f}% "
        f"(+/- {cv_scores.std() * 100:.2f})")
    log("\nConfusion matrix (rows = actual, columns = predicted)")
    log(f"                 pred not-spam   pred spam")
    log(f"actual not-spam  {tn:>13}   {fp:>9}")
    log(f"actual spam      {fn:>13}   {tp:>9}")
    log("\n" + classification_report(y_test, predictions, target_names=["not spam", "spam"],
                                     zero_division=0))

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_raw": int(total_raw),
        "n_total": int(len(df)),
        "n_spam": int((df.label == "spam").sum()),
        "n_ham": int((df.label == "ham").sum()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(len(pipeline.named_steps["tfidf"].vocabulary_)),
        "top_spam_words": top_words(pipeline, X_train, y_train, "spam"),
        "top_ham_words": top_words(pipeline, X_train, y_train, "ham"),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log(f"Model saved to {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    try:
        train_and_save()
    except RuntimeError as error:
        print("\nERROR:", error)
        raise SystemExit(1)

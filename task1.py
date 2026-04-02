from __future__ import annotations

import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from settings import (
    FIGURE_DPI,
    OUTPUT_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    SCRAPED_JSON,
    TEST_SIZE,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
)

#  Constants

AG_LABEL_MAP: dict[int, str] = {
    0: "world",
    1: "sports",
    2: "business",
    3: "science/technology",
}

SPHERES = sorted(AG_LABEL_MAP.values())

# cap training size to keep runtime reasonable
MAX_TRAIN_SAMPLES = 20_000


#  Labeled dataset

def load_ag_news() -> tuple[list[str], list[str]]:
    ds = load_dataset("fancyzhx/ag_news", split="train")
    texts: list[str] = []
    labels: list[str] = []

    for row in ds:
        sphere = AG_LABEL_MAP[row["label"]]
        text = row["text"].strip()
        if len(text) < 20:
            continue
        texts.append(text)
        labels.append(sphere)
        if len(texts) >= MAX_TRAIN_SAMPLES:
            break

    print(f"[OK] Loaded AG News: {len(texts)} articles, {len(set(labels))} categories")
    dist = Counter(labels)
    for sphere, cnt in sorted(dist.items()):
        print(f"  {sphere}: {cnt}")

    return texts, labels


#  Scraped data

def load_scraped_data() -> pd.DataFrame:
    with open(SCRAPED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"[OK] Loaded {len(df)} scraped articles from {SCRAPED_JSON}")
    return df


#  Training

def train_and_evaluate(
    texts: list[str], labels: list[str],
) -> tuple[LinearSVC, TfidfVectorizer, float, str]:

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
        max_df=TFIDF_MAX_DF,
        min_df=3,
        ngram_range=(1, 2),
    )

    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"[INFO] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    clf = LinearSVC(random_state=RANDOM_STATE, max_iter=3000)
    clf.fit(X_train, y_train)
    print("[OK] LinearSVC trained")

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"[OK] Accuracy: {accuracy:.4f}")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=SPHERES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SPHERES)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title("Confusion Matrix - AG News (supervised)")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "01_confusion_matrix.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"[OK] Saved {path}")

    eval_path = os.path.join(OUTPUT_DIR, "01_model_evaluation.csv")
    eval_df = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    ).T
    eval_df.to_csv(eval_path, sep=";", encoding="utf-8")
    print(f"[OK] Saved {eval_path}")

    return clf, vectorizer, accuracy, report


#  Apply to scraped news

def classify_scraped_news(
    clf: LinearSVC, vectorizer: TfidfVectorizer, df: pd.DataFrame,
) -> pd.DataFrame:
    X_scraped = vectorizer.transform(df["text"])
    df = df.copy()
    df["predicted_sphere"] = clf.predict(X_scraped)

    dist = df["predicted_sphere"].value_counts()

    dist_path = os.path.join(OUTPUT_DIR, "01_sphere_distribution.csv")
    dist.reset_index().rename(
        columns={"index": "sphere", "predicted_sphere": "sphere", "count": "count"},
    ).to_csv(dist_path, sep=";", index=False, encoding="utf-8")
    print(f"[OK] Saved {dist_path}")

    # bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    dist.sort_index().plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Scraped BBC News - Predicted Spheres")
    ax.set_xlabel("Sphere")
    ax.set_ylabel("Count")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "01_scraped_distribution.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"[OK] Saved {path}")

    csv_path = os.path.join(OUTPUT_DIR, "01_classified_news.csv")
    df[["title", "predicted_sphere", "text"]].to_csv(
        csv_path, sep=";", index=False, encoding="utf-8",
    )
    print(f"[OK] Saved {csv_path}")

    return df


#  Entry

def main() -> None:
    texts, labels = load_ag_news()
    clf, vectorizer, accuracy, _ = train_and_evaluate(texts, labels)

    if accuracy >= 0.60:
        print(f"[OK] Accuracy {accuracy:.2%} >= 60%")
    else:
        print(f"[WARN] Accuracy {accuracy:.2%} < 60%")

    df_scraped = load_scraped_data()
    classify_scraped_news(clf, vectorizer, df_scraped)


if __name__ == "__main__":
    main()

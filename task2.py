from __future__ import annotations

import json
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
from nltk import FreqDist
from nltk.draw.dispersion import dispersion_plot
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer

from settings import (
    BIGRAM_TOP_N,
    DISPERSION_TARGETS_N,
    FIGURE_DPI,
    OUTPUT_DIR,
    PLOTS_DIR,
    SCRAPED_JSON,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TOP_N_WORDS,
)

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))


#  Helpers

def load_texts() -> list[str]:
    with open(SCRAPED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    print(f"[OK] Loaded {len(texts)} texts for frequency analysis")
    return texts


def clean_tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    # skip noise from abbreviations
    return [w for w in tokens if len(w) > 2 and w not in STOP_WORDS]


#  TF-IDF

def analyze_tfidf(texts: list[str]) -> None:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
        max_df=TFIDF_MAX_DF,
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # average TF-IDF score per word across all documents
    mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = mean_tfidf.argsort()[::-1][:TOP_N_WORDS]

    tfidf_df = pd.DataFrame({
        "word": [feature_names[i] for i in top_indices],
        "mean_tfidf": [mean_tfidf[i] for i in top_indices],
    })
    csv_path = os.path.join(OUTPUT_DIR, "02_tfidf_top_words.csv")
    tfidf_df.to_csv(csv_path, sep=";", index=False, encoding="utf-8")
    print(f"[OK] Saved {csv_path}")

    # bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    words = tfidf_df["word"].tolist()
    scores = tfidf_df["mean_tfidf"].tolist()
    ax.barh(words[::-1], scores[::-1], color="steelblue", edgecolor="black")
    ax.set_xlabel("Mean TF-IDF Score")
    ax.set_title(f"Top-{TOP_N_WORDS} Words by Mean TF-IDF")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "02_tfidf_top_words.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"[OK] Saved {path}")


#  Lexical dispersion

def analyze_dispersion(all_tokens: list[str]) -> None:
    freq = Counter(all_tokens)
    # pick top N most frequent words as dispersion targets
    targets = [word for word, _ in freq.most_common(DISPERSION_TARGETS_N)]
    print(f"\n[INFO] Lexical dispersion targets: {targets}")

    plt.figure(figsize=(14, 6))
    dispersion_plot(all_tokens, targets, title="Lexical Dispersion Plot")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "02_lexical_dispersion.png")
    plt.savefig(path, dpi=FIGURE_DPI)
    plt.close()
    print(f"[OK] Saved {path}")


#  Word length distribution

def analyze_word_lengths(all_tokens: list[str]) -> None:
    lengths = [len(w) for w in all_tokens]
    length_freq = Counter(lengths)

    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_lengths = sorted(length_freq.keys())
    counts = [length_freq[l] for l in sorted_lengths]
    ax.bar(sorted_lengths, counts, color="coral", edgecolor="black")
    ax.set_xlabel("Word Length (characters)")
    ax.set_ylabel("Frequency")
    ax.set_title("Word Length Distribution")
    ax.set_xticks(sorted_lengths)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "02_word_length_distribution.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"[OK] Saved {path}")


#  Bigram analysis

def analyze_bigrams(all_tokens: list[str]) -> None:
    bigram_list = list(ngrams(all_tokens, 2))
    bigram_freq = Counter(bigram_list)
    top_bigrams = bigram_freq.most_common(BIGRAM_TOP_N)

    bigram_df = pd.DataFrame([
        {"word_1": b[0], "word_2": b[1], "count": c}
        for (b, c) in top_bigrams
    ])
    csv_path = os.path.join(OUTPUT_DIR, "02_bigrams.csv")
    bigram_df.to_csv(csv_path, sep=";", index=False, encoding="utf-8")
    print(f"[OK] Saved {csv_path}")

    # bar chart of top bigrams
    fig, ax = plt.subplots(figsize=(14, 7))
    labels = [f"{b[0]} {b[1]}" for b, _ in top_bigrams[:20]]
    counts = [c for _, c in top_bigrams[:20]]
    ax.barh(labels[::-1], counts[::-1], color="mediumpurple", edgecolor="black")
    ax.set_xlabel("Frequency")
    ax.set_title("Top-20 Bigrams")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "02_bigrams_bar.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"[OK] Saved {path}")

    # co-occurrence network graph
    G = nx.Graph()
    for (w1, w2), cnt in top_bigrams:
        G.add_edge(w1, w2, weight=cnt)

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, k=1.5, seed=42)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    widths = [1 + 3 * w / max_w for w in weights]

    nx.draw(
        G, pos, ax=ax,
        with_labels=True,
        node_color="skyblue",
        node_size=600,
        edge_color="gray",
        width=widths,
        font_size=8,
    )
    ax.set_title("Bigram Co-occurrence Network")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "02_bigram_network.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"[OK] Saved {path}")


#  Frequency plot

def plot_word_frequency(all_tokens: list[str]) -> None:
    freq_dist = FreqDist(all_tokens)

    fig, ax = plt.subplots(figsize=(14, 6))
    freq_dist.plot(TOP_N_WORDS, cumulative=False, show=False)
    plt.title(f"Top-{TOP_N_WORDS} Word Frequency")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "02_word_frequency.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"[OK] Saved {path}")


#  Entry

def main() -> None:
    texts = load_texts()

    all_tokens: list[str] = []
    for text in texts:
        all_tokens.extend(clean_tokenize(text))
    print(f"[INFO] Total cleaned tokens: {len(all_tokens)}, unique: {len(set(all_tokens))}")

    analyze_tfidf(texts)
    plot_word_frequency(all_tokens)
    analyze_dispersion(all_tokens)
    analyze_word_lengths(all_tokens)
    analyze_bigrams(all_tokens)


if __name__ == "__main__":
    main()

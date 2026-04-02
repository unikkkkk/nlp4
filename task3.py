from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from settings import (
    FIGURE_DPI,
    OUTPUT_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    SCRAPED_JSON,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
)


#  Data loading

def load_texts() -> list[str]:
    with open(SCRAPED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    print(f"[OK] Loaded {len(texts)} texts for unsupervised clustering")
    return texts


#  Vectorize

def vectorize(texts: list[str]) -> tuple[any, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
        max_df=TFIDF_MAX_DF,
        min_df=1,
    )
    X = vectorizer.fit_transform(texts)
    print(f"[OK] TF-IDF matrix: {X.shape}")
    return X, vectorizer


#  Optimal K

def find_optimal_k(X, k_range: range) -> tuple[list[float], list[float], int]:
    silhouette_scores: list[float] = []
    inertias: list[float] = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        silhouette_scores.append(sil)
        inertias.append(km.inertia_)
        print(f"[INFO] k={k}: silhouette={sil:.4f}, inertia={km.inertia_:.1f}")

    optimal_k = list(k_range)[np.argmax(silhouette_scores)]
    print(f"[OK] Optimal k by silhouette: {optimal_k}")
    return silhouette_scores, inertias, optimal_k


def plot_elbow_silhouette(
    k_range: range, silhouette_scores: list[float], inertias: list[float],
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(list(k_range), silhouette_scores, "o-", color="steelblue")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Silhouette Score")
    ax1.set_title("Silhouette Score vs. k")
    ax1.grid(True, alpha=0.3)

    ax2.plot(list(k_range), inertias, "o-", color="coral")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Inertia")
    ax2.set_title("Elbow Method")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "03_elbow_silhouette.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"[OK] Saved {path}")


#  Final clustering

def cluster_and_visualize(
    X, texts: list[str], n_clusters: int, vectorizer: TfidfVectorizer,
) -> None:
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10, max_iter=500)
    pred_labels = km.fit_predict(X)

    sil = silhouette_score(X, pred_labels)
    print(f"[OK] Final clustering (k={n_clusters}): silhouette={sil:.4f}")

    # build cluster topic labels from top-3 terms
    feature_names = vectorizer.get_feature_names_out()
    cluster_topics: dict[int, str] = {}
    for i, center in enumerate(km.cluster_centers_):
        top_indices = center.argsort()[::-1][:3]
        cluster_topics[i] = "_".join(feature_names[idx] for idx in top_indices)

    df = pd.DataFrame({"text": texts, "cluster": pred_labels})
    df["topic"] = df["cluster"].map(cluster_topics)

    csv_path = os.path.join(OUTPUT_DIR, "03_unsupervised_clusters.csv")
    df.to_csv(csv_path, sep=";", index=False, encoding="utf-8")
    print(f"[OK] Saved {csv_path}")

    # PCA 2D visualisation
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(X.toarray())
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(n_clusters):
        mask = pred_labels == i
        ax.scatter(
            reduced[mask, 0], reduced[mask, 1],
            s=30, color=colors[i], label=f"Cluster {i}", alpha=0.7,
        )
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title(f"KMeans Clustering (k={n_clusters}) - PCA Projection")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "03_clusters_pca.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"[OK] Saved {path}")


#  Entry

def main() -> None:
    texts = load_texts()
    X, vectorizer = vectorize(texts)

    k_range = range(2, 11)
    silhouette_scores, inertias, optimal_k = find_optimal_k(X, k_range)
    plot_elbow_silhouette(k_range, silhouette_scores, inertias)

    print(f"[INFO] Using k={optimal_k} (optimal by silhouette)")
    cluster_and_visualize(X, texts, optimal_k, vectorizer)


if __name__ == "__main__":
    main()

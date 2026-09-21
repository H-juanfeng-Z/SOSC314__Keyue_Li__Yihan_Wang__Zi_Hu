import os
import re
import html
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from itertools import combinations


# ============================================================
# 1. Configuration
# ============================================================

INPUT_FILE = "questions_clean.parquet"

N_SAMPLE = 20000

RANDOM_STATE = 42

# ---- K from 2 to 40 ----
K_VALUES = range(2, 41)

TOP_N_WORDS = 15


# ============================================================
# 2. Load data
# ============================================================

print("Loading data...")

df = pd.read_parquet(INPUT_FILE)

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())


df["title"] = df["title"].fillna("")
df["body_html"] = df["body_html"].fillna("")


# ============================================================
# 3. Combine Title + Body
# ============================================================

df["text"] = (
    df["title"].astype(str)
    + " "
    + df["body_html"].astype(str)
)


df = df[df["text"].str.strip() != ""].copy()

print("After removing empty text:", len(df))


# ============================================================
# 4. Sample
# ============================================================

if N_SAMPLE is not None and len(df) > N_SAMPLE:

    df_sample = df.sample(
        n=N_SAMPLE,
        random_state=RANDOM_STATE
    ).copy()

else:

    df_sample = df.copy()


print("Sample size:", len(df_sample))


# ============================================================
# 5. Natural-language text preprocessing
# ============================================================

def clean_text(text):

    text = str(text)

    # 1. Decode HTML entities
    text = html.unescape(text)

    # 2. Remove <pre>...</pre> code blocks
    text = re.sub(
        r"<pre\b[^>]*>.*?</pre>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # 3. Remove <code>...</code> blocks
    text = re.sub(
        r"<code\b[^>]*>.*?</code>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # 4. Remove remaining HTML tags
    text = BeautifulSoup(
        text,
        "html.parser"
    ).get_text(" ")

    # 5. Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # 6. Lowercase
    text = text.lower()

    # 7. Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


print("Cleaning text...")

df_sample["clean_text"] = (
    df_sample["text"]
    .map(clean_text)
)


df_sample = df_sample[
    df_sample["clean_text"].str.strip() != ""
].copy()

print(
    "After preprocessing and removing empty text:",
    len(df_sample)
)


# ============================================================
# 6. TF-IDF
# ============================================================

print("\nBuilding TF-IDF matrix...")


vectorizer = TfidfVectorizer(

    ngram_range=(1, 2),

    min_df=10,

    max_df=0.8,

    max_features=50000,

    stop_words="english",

    sublinear_tf=True
)


X = vectorizer.fit_transform(
    df_sample["clean_text"]
)


print("TF-IDF matrix shape:", X.shape)

print(
    "Vocabulary size:",
    len(vectorizer.get_feature_names_out())
)


feature_names = vectorizer.get_feature_names_out()


# ============================================================
# 7. Run NMF for K = 2 to 40
# ============================================================

eval_rows = []

nmf_cache = {}

for K in K_VALUES:

    print("\n" + "=" * 70)
    print(f"Running NMF with K = {K}")
    print("=" * 70)

    nmf = NMF(
        n_components=K,
        init="nndsvda",
        random_state=RANDOM_STATE,
        max_iter=300
    )

    W = nmf.fit_transform(X)

    H = nmf.components_

    nmf_cache[K] = {
        "W": W,
        "H": H,
        "recon_err": nmf.reconstruction_err_,
    }

    # ========================================================
    # 7A. Print top words for each topic
    # ========================================================

    for topic_idx in range(K):

        top_indices = (
            H[topic_idx]
            .argsort()[::-1][:TOP_N_WORDS]
        )

        top_words = [
            feature_names[i]
            for i in top_indices
        ]

        print(f"\nTopic {topic_idx}:")
        print(", ".join(top_words))

    # ========================================================
    # 7B. Document-topic matrix
    # ========================================================

    topic_proportions = W / (
        W.sum(axis=1, keepdims=True) + 1e-12
    )

    dominant_topic = topic_proportions.argmax(axis=1)

    dominant_probability = topic_proportions.max(axis=1)

    # ========================================================
    # 7C. Evaluation metrics
    # ========================================================

    topic_words = {}
    for t in range(K):
        top_idx = H[t].argsort()[::-1][:TOP_N_WORDS]
        topic_words[t] = set(feature_names[i] for i in top_idx)

    overlaps = []
    for t1, t2 in combinations(range(K), 2):
        inter = topic_words[t1] & topic_words[t2]
        union = topic_words[t1] | topic_words[t2]
        jaccard = len(inter) / len(union) if union else 0
        overlaps.append(jaccard)

    mean_overlap_count = np.mean([
        len(topic_words[t1] & topic_words[t2])
        for t1, t2 in combinations(range(K), 2)
    ])
    max_overlap_count = np.max([
        len(topic_words[t1] & topic_words[t2])
        for t1, t2 in combinations(range(K), 2)
    ])
    mean_jaccard = np.mean(overlaps)

    mean_dom = dominant_probability.mean()
    std_dom = dominant_probability.std()
    frac_ambiguous = (dominant_probability < 0.5).mean()

    topic_counts = np.bincount(dominant_topic, minlength=K)
    topic_props_size = topic_counts / topic_counts.sum()
    entropy = -np.sum(topic_props_size * np.log(topic_props_size + 1e-12))
    max_entropy = np.log(K)
    norm_entropy = entropy / max_entropy

    num_topics_above_5pct = (topic_props_size >= 0.05).sum()
    min_topic_prop = topic_props_size.min()
    max_topic_prop = topic_props_size.max()

    eval_rows.append({
        "K": K,
        "recon_err": nmf.reconstruction_err_,
        "mean_dom_prob": mean_dom,
        "std_dom_prob": std_dom,
        "frac_ambiguous": frac_ambiguous,
        "topic_size_entropy_norm": norm_entropy,
        "num_topics_above_5pct": num_topics_above_5pct,
        "min_topic_prop": min_topic_prop,
        "max_topic_prop": max_topic_prop,
        "mean_overlap_count": mean_overlap_count,
        "max_overlap_count": max_overlap_count,
        "mean_jaccard": mean_jaccard,
    })


# ============================================================
# 8. Evaluation summary
# ============================================================

eval_df = pd.DataFrame(eval_rows)

print("\n" + "=" * 70)
print("EVALUATION SUMMARY (K = 2 to 40)")
print("=" * 70)

print(eval_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ============================================================
# 9. Best K by different criteria
# ============================================================

print("\n" + "=" * 70)
print("BEST K BY DIFFERENT CRITERIA")
print("=" * 70)

best_recon = eval_df.loc[eval_df["recon_err"].idxmin()]
print(f"\n1. Lowest reconstruction error: K={int(best_recon['K'])}, "
      f"recon_err={best_recon['recon_err']:.3f}")

best_dom = eval_df.loc[eval_df["mean_dom_prob"].idxmax()]
print(f"2. Highest mean dominant prob: K={int(best_dom['K'])}, "
      f"mean_dom_prob={best_dom['mean_dom_prob']:.3f}")

best_amb = eval_df.loc[eval_df["frac_ambiguous"].idxmin()]
print(f"3. Lowest frac ambiguous: K={int(best_amb['K'])}, "
      f"frac_ambiguous={best_amb['frac_ambiguous']:.3f}")

best_ent = eval_df.loc[eval_df["topic_size_entropy_norm"].idxmax()]
print(f"4. Highest topic size entropy: K={int(best_ent['K'])}, "
      f"entropy={best_ent['topic_size_entropy_norm']:.3f}")

best_ov = eval_df.loc[eval_df["mean_overlap_count"].idxmin()]
print(f"5. Lowest mean overlap: K={int(best_ov['K'])}, "
      f"mean_overlap={best_ov['mean_overlap_count']:.3f}")

best_max_ov = eval_df.loc[eval_df["max_overlap_count"].idxmin()]
print(f"6. Lowest max overlap: K={int(best_max_ov['K'])}, "
      f"max_overlap={best_max_ov['max_overlap_count']}")

eval_df["topic_coverage"] = eval_df["num_topics_above_5pct"] / eval_df["K"]
best_cov = eval_df.loc[eval_df["topic_coverage"].idxmax()]
print(f"7. Best topic coverage: K={int(best_cov['K'])}, "
      f"coverage={best_cov['topic_coverage']:.3f}")

eval_df_sorted = eval_df.sort_values("K").reset_index(drop=True)
eval_df_sorted["recon_diff"] = eval_df_sorted["recon_err"].diff()
eval_df_sorted["recon_diff2"] = eval_df_sorted["recon_diff"].diff()

if len(eval_df_sorted) > 2:
    elbow_idx = eval_df_sorted["recon_diff2"].idxmax()
    if elbow_idx is not None and not pd.isna(elbow_idx):
        best_elbow = eval_df_sorted.loc[elbow_idx]
        print(f"8. Elbow method: K={int(best_elbow['K'])}, "
              f"recon_err={best_elbow['recon_err']:.3f}")

eval_df["rank_recon"] = eval_df["recon_err"].rank()
eval_df["rank_dom"] = eval_df["mean_dom_prob"].rank(ascending=False)
eval_df["rank_amb"] = eval_df["frac_ambiguous"].rank()
eval_df["rank_ent"] = eval_df["topic_size_entropy_norm"].rank(ascending=False)
eval_df["rank_ov"] = eval_df["mean_overlap_count"].rank()
eval_df["rank_max_ov"] = eval_df["max_overlap_count"].rank()

eval_df["composite_rank"] = (
    eval_df["rank_recon"] +
    eval_df["rank_dom"] +
    eval_df["rank_amb"] +
    eval_df["rank_ent"] +
    eval_df["rank_ov"] +
    eval_df["rank_max_ov"]
) / 6

best_composite = eval_df.loc[eval_df["composite_rank"].idxmin()]
print(f"\n9. Best composite rank: K={int(best_composite['K'])}, "
      f"composite_rank={best_composite['composite_rank']:.2f}")


# ============================================================
# 10. Final output
# ============================================================

print("\n" + "=" * 70)
print("Finished.")
print("=" * 70)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

axes[0, 0].plot(eval_df["K"], eval_df["recon_err"], marker="o")
axes[0, 0].set_title("Reconstruction Error")
axes[0, 0].set_xlabel("K")

axes[0, 1].plot(eval_df["K"], eval_df["mean_dom_prob"], marker="o")
axes[0, 1].set_title("Mean Dominant Probability")
axes[0, 1].set_xlabel("K")

axes[0, 2].plot(eval_df["K"], eval_df["frac_ambiguous"], marker="o")
axes[0, 2].set_title("Fraction Ambiguous (dom < 0.5)")
axes[0, 2].set_xlabel("K")

axes[1, 0].plot(eval_df["K"], eval_df["topic_size_entropy_norm"], marker="o")
axes[1, 0].set_title("Normalized Topic Size Entropy")
axes[1, 0].set_xlabel("K")

axes[1, 1].plot(eval_df["K"], eval_df["mean_overlap_count"], marker="o")
axes[1, 1].set_title("Mean Topic Overlap Count")
axes[1, 1].set_xlabel("K")

axes[1, 2].plot(eval_df["K"], eval_df["num_topics_above_5pct"], marker="o")
axes[1, 2].set_title("Num Topics >= 5%")
axes[1, 2].set_xlabel("K")

plt.tight_layout()
plt.show()
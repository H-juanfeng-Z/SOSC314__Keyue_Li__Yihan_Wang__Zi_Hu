import os
import re
import html
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# ============================================================
# Configuration
# ============================================================
INPUT_FILE = "questions_clean.parquet"
N_SAMPLE = 20000
RANDOM_STATE = 42

# Number of main topics for the first-stage NMF
K_MAIN = 8

# Which main topics to subdivide, and how many sub-topics each should have
SUBDIVIDE = {
    3: 2,   # OOP vs GUI
    5: 2,   # numpy vs matplotlib
    6: 2,   # install vs file
    7: 3,   # API vs scraping vs Django views
}

# Number of top words to print for each sub-topic
TOP_N_WORDS = 12


# ============================================================
# 1. Load data + baseline preprocessing
# ============================================================
# Load the cleaned parquet file
df = pd.read_parquet(INPUT_FILE)

# Fill missing values so string concatenation does not produce "nan"
df["title"] = df["title"].fillna("")
df["body_html"] = df["body_html"].fillna("")

# Combine title and body into a single text column
df["text"] = df["title"].astype(str) + " " + df["body_html"].astype(str)

# Drop rows where the combined text is empty
df = df[df["text"].str.strip() != ""].copy()

# Randomly sample N_SAMPLE rows if the dataset is larger than N_SAMPLE
if len(df) > N_SAMPLE:
    df_sample = df.sample(n=N_SAMPLE, random_state=RANDOM_STATE).copy()
else:
    df_sample = df.copy()


def clean_baseline(text):
    """
    Basic text cleaning:
    - unescape HTML entities
    - strip HTML tags
    - lowercase
    - replace URLs with a placeholder
    - collapse whitespace
    """
    text = str(text)
    text = html.unescape(text)
    text = BeautifulSoup(text, "lxml").get_text()
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Apply the cleaning function to every row
df_sample["clean_text"] = df_sample["text"].map(clean_baseline)

# Drop rows that became empty after cleaning
df_sample = df_sample[df_sample["clean_text"].str.strip() != ""].copy()
df_sample = df_sample.reset_index(drop=True)
print("Sample size:", len(df_sample))


# ============================================================
# 2. Main NMF (K=8) to discover the main topics
# ============================================================
# Convert cleaned text into a TF-IDF matrix
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), min_df=10, max_df=0.8,
    max_features=50000, stop_words="english", sublinear_tf=True
)
X = vectorizer.fit_transform(df_sample["clean_text"])
feature_names = vectorizer.get_feature_names_out()

# Run NMF to factorize the TF-IDF matrix into W (doc-topic) and H (topic-word)
nmf = NMF(n_components=K_MAIN, init="nndsvda",
          random_state=RANDOM_STATE, max_iter=300)
W = nmf.fit_transform(X)
H = nmf.components_

# Normalize W so each row sums to 1 (topic proportions per document)
topic_props = W / (W.sum(axis=1, keepdims=True) + 1e-12)

# Assign each document to its dominant main topic
dominant_topic = topic_props.argmax(axis=1)

# Print the top words for each main topic
print("\nMain topics (K=8):")
for t in range(K_MAIN):
    top_idx = H[t].argsort()[::-1][:15]
    words = [feature_names[i] for i in top_idx]
    print(f"  Topic {t}: {', '.join(words)}")


# ============================================================
# 3. Subdivide selected main topics (3, 5, 6, 7)
# ============================================================
# Store the main topic assignment on the dataframe
df_sample["main_topic"] = dominant_topic

# Dictionary mapping row index -> sub-topic id (only for subdivided rows)
sub_assignments = {}

for main_topic, K_sub in SUBDIVIDE.items():
    # Select only rows belonging to this main topic
    mask = dominant_topic == main_topic
    sub_texts = df_sample.loc[mask, "clean_text"]

    # Build a separate TF-IDF matrix for this subset
    vec_sub = TfidfVectorizer(
        ngram_range=(1, 2), min_df=5, max_df=0.9,
        max_features=20000, stop_words="english", sublinear_tf=True
    )
    X_sub = vec_sub.fit_transform(sub_texts)
    feat_sub = vec_sub.get_feature_names_out()

    # Run NMF again on the subset to split it into K_sub sub-topics
    nmf_sub = NMF(n_components=K_sub, init="nndsvda",
                  random_state=RANDOM_STATE, max_iter=300)
    W_sub = nmf_sub.fit_transform(X_sub)
    H_sub = nmf_sub.components_

    # Normalize W_sub and assign each document to its dominant sub-topic
    sub_props = W_sub / (W_sub.sum(axis=1, keepdims=True) + 1e-12)
    sub_dominant = sub_props.argmax(axis=1)

    # Print the top words for each sub-topic
    print(f"\nSub-topics of main Topic {main_topic} (K={K_sub}):")
    for st in range(K_sub):
        top_idx = H_sub[st].argsort()[::-1][:TOP_N_WORDS]
        words = [feat_sub[i] for i in top_idx]
        print(f"  Sub {st}: {', '.join(words)}")

    # Record sub-topic assignments back into the global dictionary
    idx_in_full = np.where(mask)[0]
    for i, sub_t in zip(idx_in_full, sub_dominant):
        sub_assignments[i] = sub_t


# ============================================================
# 4. Flatten main + sub topics into 13 final types
# ============================================================
# Mapping from (main_topic, sub_topic) -> flat_id
# Use None as the sub-topic key when the main topic is not subdivided
FLAT_LABELS = {
    0: {None: 1},   # Python basics
    1: {None: 2},   # errors/environment
    2: {None: 3},   # Django models
    3: {0: 4, 1: 5},   # OOP, Tkinter GUI
    4: {None: 6},   # pandas
    5: {0: 7, 1: 8},   # numpy, matplotlib
    6: {0: 9, 1: 10},  # install/package, file/os/path
    7: {0: 11, 1: 12, 2: 13},  # API/server, scraping, Django views/urls
}

# Human-readable names for the 13 final types
FLAT_NAMES = {
    1: "Python basics",
    2: "errors/environment",
    3: "Django models",
    4: "OOP",
    5: "Tkinter GUI",
    6: "pandas",
    7: "numpy",
    8: "matplotlib",
    9: "install/package",
    10: "file/os/path",
    11: "API/server",
    12: "BeautifulSoup/scraping",
    13: "Django views/urls",
}

# Assign the final flat id to every row
final_ids = np.zeros(len(df_sample), dtype=int)

for i in range(len(df_sample)):
    mt = df_sample.iloc[i]["main_topic"]
    if i in sub_assignments:
        # Row belongs to a subdivided main topic
        st = sub_assignments[i]
        flat_id = FLAT_LABELS[mt][st]
    else:
        # Row belongs to a main topic that was not subdivided
        flat_id = FLAT_LABELS[mt][None]
    final_ids[i] = flat_id

df_sample["flat_id"] = final_ids
df_sample["flat_name"] = df_sample["flat_id"].map(FLAT_NAMES)

print("\nColumns after flattening:", df_sample.columns.tolist())
print("flat_name sample:")
print(df_sample["flat_name"].head())


# ============================================================
# 7. Overlap analysis: three figures (display only, no export)
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ---- Figure 3: similarity between the 13 types ----
# Build a global TF-IDF matrix over all documents
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    min_df=5,
)
X_all = vectorizer.fit_transform(df_sample["clean_text"])

# Get the sorted list of type ids that actually appear
type_ids = sorted(df_sample["flat_id"].unique())

# Compute a mean TF-IDF vector for each type
type_vectors = {}
for fid in type_ids:
    mask = (df_sample["flat_id"] == fid).values  # convert to numpy for indexing
    type_vectors[fid] = np.asarray(X_all[mask].mean(axis=0)).ravel()

# Stack the per-type vectors into a matrix and compute pairwise cosine similarity
type_matrix = np.array([type_vectors[fid] for fid in type_ids])
sim13 = cosine_similarity(type_matrix)

# Human-readable labels for the heatmap axes
labels = [FLAT_NAMES[f] for f in type_ids]

# Plot the 13x13 similarity heatmap
plt.figure(figsize=(13, 11))
sns.heatmap(
    sim13, annot=True, fmt=".2f", cmap="Reds",
    xticklabels=labels,
    yticklabels=labels,
    vmin=0, vmax=1,
)
plt.title("Type similarity (cosine on mean TF-IDF)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
plt.close()
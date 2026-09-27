import os
import re
import html
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

INPUT_FILE = "questions_clean.parquet"
N_SAMPLE = 20000
RANDOM_STATE = 42

K_MAIN = 8
SUBDIVIDE = {
    3: 2,   # OOP vs GUI
    5: 2,   # numpy vs matplotlib
    6: 2,   # install vs file
    7: 3,   # API vs scraping vs Django views
}

TOP_N_WORDS = 12


# ============================================================
# 1. Load + baseline preprocessing
# ============================================================
df = pd.read_parquet(INPUT_FILE)
df["title"] = df["title"].fillna("")
df["body_html"] = df["body_html"].fillna("")
df["text"] = df["title"].astype(str) + " " + df["body_html"].astype(str)
df = df[df["text"].str.strip() != ""].copy()

if len(df) > N_SAMPLE:
    df_sample = df.sample(n=N_SAMPLE, random_state=RANDOM_STATE).copy()
else:
    df_sample = df.copy()


def clean_baseline(text):
    text = str(text)
    text = html.unescape(text)
    text = BeautifulSoup(text, "lxml").get_text()
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df_sample["clean_text"] = df_sample["text"].map(clean_baseline)
df_sample = df_sample[df_sample["clean_text"].str.strip() != ""].copy()
df_sample = df_sample.reset_index(drop=True)
print("Sample size:", len(df_sample))


# ============================================================
# 2. Main NMF (K=8)
# ============================================================
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), min_df=10, max_df=0.8,
    max_features=50000, stop_words="english", sublinear_tf=True
)
X = vectorizer.fit_transform(df_sample["clean_text"])
feature_names = vectorizer.get_feature_names_out()

nmf = NMF(n_components=K_MAIN, init="nndsvda",
          random_state=RANDOM_STATE, max_iter=300)
W = nmf.fit_transform(X)
H = nmf.components_

topic_props = W / (W.sum(axis=1, keepdims=True) + 1e-12)
dominant_topic = topic_props.argmax(axis=1)

print("\nMain topics (K=8):")
for t in range(K_MAIN):
    top_idx = H[t].argsort()[::-1][:15]
    words = [feature_names[i] for i in top_idx]
    print(f"  Topic {t}: {', '.join(words)}")


# ============================================================
# 3. Subdivide Topic 3/5/6/7
# ============================================================
df_sample["main_topic"] = dominant_topic

sub_assignments = {}

for main_topic, K_sub in SUBDIVIDE.items():
    mask = dominant_topic == main_topic
    sub_texts = df_sample.loc[mask, "clean_text"]

    vec_sub = TfidfVectorizer(
        ngram_range=(1, 2), min_df=5, max_df=0.9,
        max_features=20000, stop_words="english", sublinear_tf=True
    )
    X_sub = vec_sub.fit_transform(sub_texts)
    feat_sub = vec_sub.get_feature_names_out()

    nmf_sub = NMF(n_components=K_sub, init="nndsvda",
                  random_state=RANDOM_STATE, max_iter=300)
    W_sub = nmf_sub.fit_transform(X_sub)
    H_sub = nmf_sub.components_

    sub_props = W_sub / (W_sub.sum(axis=1, keepdims=True) + 1e-12)
    sub_dominant = sub_props.argmax(axis=1)

    print(f"\nSub-topics of main Topic {main_topic} (K={K_sub}):")
    for st in range(K_sub):
        top_idx = H_sub[st].argsort()[::-1][:TOP_N_WORDS]
        words = [feat_sub[i] for i in top_idx]
        print(f"  Sub {st}: {', '.join(words)}")

    idx_in_full = np.where(mask)[0]
    for i, sub_t in zip(idx_in_full, sub_dominant):
        sub_assignments[i] = sub_t


# ============================================================
# 4. Flatten to 13 types
# ============================================================
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

final_ids = np.zeros(len(df_sample), dtype=int)

for i in range(len(df_sample)):
    mt = df_sample.iloc[i]["main_topic"]
    if i in sub_assignments:
        st = sub_assignments[i]
        flat_id = FLAT_LABELS[mt][st]
    else:
        flat_id = FLAT_LABELS[mt][None]
    final_ids[i] = flat_id

df_sample["flat_id"] = final_ids
df_sample["flat_name"] = df_sample["flat_id"].map(FLAT_NAMES)

print("\nColumns after flattening:", df_sample.columns.tolist())
print("flat_name sample:")
print(df_sample["flat_name"].head())




# ============================================================
# 5. Output: question_id + title + type_id + type_name
# ============================================================
output_cols = [
    "question_id",
    "title",
    "flat_id",
    "flat_name",
]

out_df = df_sample[output_cols].copy()
out_df = out_df.rename(columns={
    "flat_id": "type_id",
    "flat_name": "type_name",
})

out_df = out_df.sort_values("type_id")

out_path = "questions_with_types.csv"
out_df.to_csv(out_path, index=False)

print(f"\nSaved to {out_path}")
print("Shape:", out_df.shape)
print("\nType distribution:")
print(out_df["type_name"].value_counts().sort_index())
print("\nPreview:")
print(out_df.head(10))
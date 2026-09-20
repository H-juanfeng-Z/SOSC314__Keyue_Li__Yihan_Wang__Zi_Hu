import os
import re
import html
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# 1. Configuration
# ============================================================

INPUT_FILE = "questions_clean.parquet"

OUTPUT_DIR = "doc2vec_kmeans_results"

# Start with 20k for exploratory analysis
N_SAMPLE = 20000

RANDOM_STATE = 42

# Compare different numbers of clusters
K_VALUES = range(4, 11)

# Doc2Vec parameters
VECTOR_SIZE = 100
WINDOW = 8
MIN_COUNT = 5
EPOCHS = 20
NEGATIVE = 10
WORKERS = 4

# Number of representative questions per cluster
TOP_N_DOCUMENTS = 10


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. Load data
# ============================================================

print("Loading data...")

df = pd.read_parquet(
    INPUT_FILE
)

print(
    "Original shape:",
    df.shape
)

print(
    "Columns:",
    df.columns.tolist()
)


# Make sure title/body are not missing
df["title"] = (
    df["title"]
    .fillna("")
)

df["body_html"] = (
    df["body_html"]
    .fillna("")
)


# ============================================================
# 3. Combine Title + Body
# ============================================================

df["text"] = (
    df["title"].astype(str)
    + " "
    + df["body_html"].astype(str)
)


# Remove completely empty questions

df = df[
    df["text"].str.strip() != ""
].copy()


print(
    "After removing empty text:",
    len(df)
)


# ============================================================
# 4. Sample for exploratory analysis
# ============================================================

if (
    N_SAMPLE is not None
    and len(df) > N_SAMPLE
):

    df_sample = df.sample(
        n=N_SAMPLE,
        random_state=RANDOM_STATE
    ).copy()

else:

    df_sample = df.copy()


print(
    "Sample size:",
    len(df_sample)
)


# ============================================================
# 5. Preprocessing for Doc2Vec
# ============================================================

def clean_text(text):

    text = str(text)


    # --------------------------------------------------------
    # 1. Decode HTML entities
    # --------------------------------------------------------

    text = html.unescape(
        text
    )


    # --------------------------------------------------------
    # 2. Remove <pre>...</pre> code blocks
    # --------------------------------------------------------
    # We want the natural-language question content,
    # rather than programming code itself.
    # --------------------------------------------------------

    text = re.sub(
        r"<pre\b[^>]*>.*?</pre>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )


    # --------------------------------------------------------
    # 3. Remove <code>...</code>
    # --------------------------------------------------------
    # This also removes inline code.
    # --------------------------------------------------------

    text = re.sub(
        r"<code\b[^>]*>.*?</code>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )


    # --------------------------------------------------------
    # 4. Remove remaining HTML
    # --------------------------------------------------------

    text = BeautifulSoup(
        text,
        "html.parser"
    ).get_text(" ")


    # --------------------------------------------------------
    # 5. Remove URLs
    # --------------------------------------------------------

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )


    # --------------------------------------------------------
    # 6. Lowercase
    # --------------------------------------------------------

    text = text.lower()


    # --------------------------------------------------------
    # 7. Normalize whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()


    return text


print(
    "Cleaning text..."
)


df_sample["clean_text"] = (
    df_sample["text"]
    .map(clean_text)
)


# Remove empty documents after preprocessing

df_sample = df_sample[
    df_sample["clean_text"].str.strip() != ""
].copy()


print(
    "After preprocessing:",
    len(df_sample)
)


# ============================================================
# 6. Tokenization
# ============================================================
#
# Important:
#
# Unlike TF-IDF, we do NOT use sklearn's stop_words="english".
#
# We also do not use stemming or lemmatization.
#
# The goal is to preserve the natural language structure
# while keeping technical terms reasonably intact.
# ============================================================

def tokenize(text):

    tokens = re.findall(
        r"[a-zA-Z]+(?:'[a-zA-Z]+)?"
        r"|\d+(?:\.\d+)?",
        text
    )

    return tokens


print(
    "Tokenizing..."
)


df_sample["tokens"] = (
    df_sample["clean_text"]
    .map(tokenize)
)


# Remove documents with no tokens

df_sample = df_sample[
    df_sample["tokens"].map(len) > 0
].copy()


print(
    "Documents with valid tokens:",
    len(df_sample)
)


# ============================================================
# 7. Create TaggedDocument objects
# ============================================================

print(
    "Creating TaggedDocument objects..."
)


tagged_documents = [

    TaggedDocument(
        words=row["tokens"],
        tags=[str(row["question_id"])]
    )

    for _, row in df_sample.iterrows()
]


print(
    "Number of tagged documents:",
    len(tagged_documents)
)


# ============================================================
# 8. Train Doc2Vec
# ============================================================

print(
    "\nTraining Doc2Vec..."
)


doc2vec_model = Doc2Vec(

    vector_size=VECTOR_SIZE,

    window=WINDOW,

    min_count=MIN_COUNT,

    workers=WORKERS,

    epochs=EPOCHS,

    negative=NEGATIVE,

    dm=0,          # PV-DBOW

    seed=RANDOM_STATE
)


# Build vocabulary

doc2vec_model.build_vocab(
    tagged_documents
)


print(
    "Vocabulary size:",
    len(doc2vec_model.wv)
)


# Train

doc2vec_model.train(
    tagged_documents,

    total_examples=doc2vec_model.corpus_count,

    epochs=doc2vec_model.epochs
)


print(
    "Doc2Vec training finished."
)


# Save model

model_path = os.path.join(
    OUTPUT_DIR,
    "doc2vec_model.model"
)

doc2vec_model.save(
    model_path
)


# ============================================================
# 9. Generate document vectors
# ============================================================

print(
    "Generating document vectors..."
)


document_vectors = np.vstack([

    doc2vec_model.dv[
        str(question_id)
    ]

    for question_id
    in df_sample["question_id"]
])


print(
    "Document vector shape:",
    document_vectors.shape
)


# Save vectors

vectors_path = os.path.join(
    OUTPUT_DIR,
    "document_vectors.npy"
)

np.save(
    vectors_path,
    document_vectors
)


# ============================================================
# 10. K-Means clustering
# ============================================================

all_results = []


for K in K_VALUES:

    print(
        "\n" + "=" * 70
    )

    print(
        f"Running K-Means with K = {K}"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # K-Means
    # --------------------------------------------------------

    kmeans = KMeans(

        n_clusters=K,

        random_state=RANDOM_STATE,

        n_init=10
    )


    cluster_labels = (
        kmeans.fit_predict(
            document_vectors
        )
    )


    # --------------------------------------------------------
    # Cluster sizes
    # --------------------------------------------------------

    cluster_sizes = np.bincount(
        cluster_labels
    )


    print(
        "Cluster sizes:",
        cluster_sizes
    )


    # --------------------------------------------------------
    # Silhouette score
    # --------------------------------------------------------

    silhouette = silhouette_score(
        document_vectors,
        cluster_labels
    )


    print(
        "Silhouette score:",
        silhouette
    )


    # ========================================================
    # 10A. Save cluster assignments
    # ========================================================

    assignments = pd.DataFrame({

        "question_id":
            df_sample[
                "question_id"
            ].values,

        "cluster":
            cluster_labels
    })


    assignments.to_csv(

        os.path.join(
            OUTPUT_DIR,
            f"kmeans_K{K}_assignments.csv"
        ),

        index=False
    )


    # ========================================================
    # 10B. Find representative questions
    # ========================================================
    #
    # We calculate distance from each question to its
    # cluster centroid.
    #
    # Smaller distance = more representative.
    # ========================================================

    representative_rows = []


    for cluster_id in range(K):

        cluster_indices = np.where(
            cluster_labels == cluster_id
        )[0]


        if len(cluster_indices) == 0:
            continue


        # Cluster centroid

        centroid = (
            kmeans.cluster_centers_[
                cluster_id
            ]
        )


        # Vectors belonging to this cluster

        cluster_vectors = (
            document_vectors[
                cluster_indices
            ]
        )


        # Euclidean distance to centroid

        distances = np.linalg.norm(
            cluster_vectors
            - centroid,
            axis=1
        )


        # Sort from most representative
        # to least representative

        sorted_positions = (
            distances.argsort()
        )


        top_positions = (
            sorted_positions[
                :TOP_N_DOCUMENTS
            ]
        )


        for rank, position in enumerate(
            top_positions,
            start=1
        ):

            original_idx = (
                cluster_indices[
                    position
                ]
            )


            row = df_sample.iloc[
                original_idx
            ]


            representative_rows.append({

                "K":
                    K,

                "cluster":
                    cluster_id,

                "rank":
                    rank,

                "distance_to_centroid":
                    distances[position],

                "question_id":
                    row["question_id"],

                "title":
                    row["title"],

                "clean_text":
                    row["clean_text"]
            })


    representative_df = pd.DataFrame(
        representative_rows
    )


    representative_df.to_csv(

        os.path.join(
            OUTPUT_DIR,
            f"kmeans_K{K}_representative_questions.csv"
        ),

        index=False
    )


    # ========================================================
    # 10C. Save cluster size + silhouette
    # ========================================================

    all_results.append({

        "K":
            K,

        "silhouette_score":
            silhouette,

        "min_cluster_size":
            cluster_sizes.min(),

        "max_cluster_size":
            cluster_sizes.max(),

        "mean_cluster_size":
            cluster_sizes.mean()
    })


# ============================================================
# 11. Save K comparison
# ============================================================

comparison = pd.DataFrame(
    all_results
)


comparison.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "kmeans_K_comparison.csv"
    ),

    index=False
)


# ============================================================
# 12. Final output
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "Finished."
)

print(
    "=" * 70
)


print(
    comparison.to_string(
        index=False
    )
)


print(
    "\nResults saved to:"
)

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)
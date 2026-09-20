## 1. `kmeans_K{K}_representative_questions.csv`


### Columns

| Column | Type | Description |
|---|---|---|
| `K` | int | Number of clusters used in K-Means (e.g., 4, 5, ..., 10). |
| `cluster` | int | Cluster ID (0 to K-1). |
| `rank` | int | Rank of the question within its cluster (1 = closest to centroid, 10 = farthest among the top 10). |
| `distance_to_centroid` | float | Euclidean distance from the question's Doc2Vec embedding to the cluster centroid. Smaller = more representative. |
| `question_id` | int | Stack Overflow question ID. |
| `title` | str | Title of the question. |
| `clean_text` | str | Preprocessed text of the question (HTML removed, lowercased, URLs replaced, whitespace normalized). |

## 2. `kmeans_K{K}_assignments.csv`


### Columns

| Column | Type | Description |
|---|---|---|
| `question_id` | int | Stack Overflow question ID. |
| `cluster` | int | Cluster ID (0 to K-1) assigned by K-Means. |

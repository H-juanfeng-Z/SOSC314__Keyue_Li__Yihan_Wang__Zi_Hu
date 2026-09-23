## 1. `nmf_K{K}_top_words.csv`


### Columns

| Column | Type | Description |
|---|---|---|
| `K` | int | Number of topics used in NMF (e.g., 2, 3, ..., 20). |
| `topic` | int | Topic ID (0 to K-1). |
| `rank` | int | Rank of the term within its topic (1 = highest weight, 15 = lowest among the top 15). |
| `term` | str | The term (unigram or bigram) extracted from the TF-IDF vocabulary. |
| `weight` | float | NMF weight of the term in this topic. Higher = more important to the topic. |
## 2. `nmf_K{K}_assignments.csv`

### Columns

| Column | Type | Description |
|---|---|---|
| `Id` | int | Stack Overflow question ID. |
| `dominant_topic` | int | Topic ID (0 to K-1) with the highest weight for this question. |
| `topic_weight` | float | The proportion of the document assigned to the dominant topic. Range: 1/K to 1.0. Higher = more confident assignment. |

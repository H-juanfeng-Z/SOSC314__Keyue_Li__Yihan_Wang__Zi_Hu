# Topic modeling

NMF and Doc2Vec/K-Means scripts were relocated from `experiments/week4/Question Type Classification/`; the verb-phrase LDA script was relocated from `scripts/`. This is a location-only reorganization of the analyses, not a new experiment.

- [Technical-topic methods](../../docs/experiments/technical_topics.md)
- [NMF example questions](../../docs/experiments/nmf_example_questions.md)
- [NMF and K-Means result snapshots](../../tables/topic_modeling/)
- [Technical-topic figures](../../figures/topic_modeling/)
- Existing LDA results remain in [tables](../../tables/) and [figures](../../figures/).

The NMF and Doc2Vec programs retain their original working-directory-relative input and output settings. Place `questions_clean.parquet` in `data/processed/`, then, from the repository root:

```sh
cd data/processed
python "../../scripts/topic_modeling/question classfication NMF.py"
```

Substitute another script filename to run the other configuration. Dependencies include pandas, NumPy, scikit-learn, BeautifulSoup, an HTML parser and a Parquet engine; Doc2Vec additionally needs gensim. Local outputs are created under this working directory, not over the committed result snapshots. The `code for NMF_nature K=2-40 figure.py` program reruns NMF and displays a plot; it is not a lightweight figure-only regeneration script.

The LDA program requires spaCy and its `en_core_web_sm` model. Its `QUESTIONS_CSV` and `OUT_DIR` settings still contain the author's original local paths and must be configured before running. They were not silently changed to another dataset in this migration. No topic naming, model settings or results were changed.

# Stack Overflow Question Classification

Location guide: [programs and run instructions](../../scripts/topic_modeling/README.md), [result tables](../../tables/topic_modeling/), [figures](../../figures/topic_modeling/), and [example questions](nmf_example_questions.md). The methodological text below is retained from the original contribution.

This project classifies Stack Overflow questions into **8 topic categories** using unsupervised topic modeling. We compare two modeling approaches (NMF vs. Doc2Vec + K-Means) under two text preprocessing strategies (baseline vs. natural language).

---

## Task

Given a dataset of Stack Overflow questions, the goal is to **group them into meaningful topic categories** without using any labels. 
The current solution classifies each question into one of **8 topics** using NMF method for baseline within K=8:

- Python basics
- Error / environment
- Django
- Object-oriented programming
- Pandas
- NumPy / Matplotlib
- File / script operations
- Web requests

---

## Data

The dataset contains **607,282 Stack Overflow questions** with 20 columns. For all experiments, we use a random sample of **20,000 questions** (`random_state=42`) to keep the analysis tractable.

---

## Preprocessing Strategies

We compared two text preprocessing strategies that differ in whether code blocks are retained. Both strategies share the same base pipeline, but diverge in one key step.

### Shared Steps (Both Strategies)

Both strategies apply the following five operations in the same order:

1. **Decode HTML entities** using `html.unescape` (e.g., `&lt;div&gt;` → `<div>`, `&amp;` → `&`).
2. **Strip HTML tags** using `BeautifulSoup`.
3. **Lowercase** all text.
4. **Replace URLs** with the placeholder `" URL "`.
5. **Normalize whitespace** by collapsing consecutive spaces into a single space.

### Baseline (keep code)

```python
def clean_text(text):
    text = str(text)
    text = html.unescape(text)
    text = BeautifulSoup(text, "lxml").get_text()
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

- Uses `lxml` as the HTML parser.
- **Keeps the content inside `<pre>` and `<code>` tags**, so code fragments remain in the text.
- **Rationale**: Code-related terms such as `django`, `charfield`, `dataframe`, `numpy`, and `traceback` are strong discriminative signals for distinguishing technology-specific topics. Retaining them allows the model to separate questions by library or framework.
- **Vocabulary size after filtering**: ~23,766 terms.

### Natural Language (remove code)

```python
def clean_text(text):
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<pre\b[^>]*>.*?</pre>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<code\b[^>]*>.*?</code>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

- **Removes `<pre>` and `<code>` blocks first** using regular expressions, then strips remaining HTML tags.
- Uses `html.parser` as the HTML parser.
- **Rationale**: Removing code blocks shifts the focus to the natural-language "intent" of the question, such as conceptual questions, or how-to requests. 
- **Vocabulary size after filtering**: ~12,283 terms.



## Methods

### Method 1: NMF (Non-negative Matrix Factorization)

**Scripts**:
- [question classfication NMF.py](../../scripts/topic_modeling/question%20classfication%20NMF.py) — baseline preprocessing (keep code)
- [question classification NMF nature.py](../../scripts/topic_modeling/question%20classification%20NMF%20nature.py) — natural-language preprocessing (remove code)

### Method 2: Doc2Vec + K-Means


**Scripts**:
- [question classification Doc2Vec baseline.py](../../scripts/topic_modeling/question%20classification%20Doc2Vec%20baseline.py) — baseline preprocessing (keep code)
- [question classification Doc2Vec nature.py](../../scripts/topic_modeling/question%20classification%20Doc2Vec%20nature.py) — natural-language preprocessing (remove code)
- [question Doc2Vec nature optimized.py](../../scripts/topic_modeling/question%20Doc2Vec%20nature%20optimized.py) — worst-case tuned configuration (PV-DM, vector size 200, 50 epochs)

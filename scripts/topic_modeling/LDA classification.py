# %%
#!/usr/bin/env python3
"""
replicate_as2013.py
===================
复现 Allamanis & Sutton (2013, MSR) ：
从 Stack Overflow 问题中提取"动词短语"(verb phrases)，
用 LDA 主题模型自动聚出"问题类型"(question types)。

原文五大类型：
  1. way of using        —— "can use", "to do"（怎么用某物）
  2. do not work         —— "not work", "fail"（代码跑不通）
  3. how/why it works    —— "understand", "explain"（不理解原理）
  4. implement something —— "to create", "to make"（不知道怎么实现）
  5. learning            —— "learn", "recommend"（求学习资源）

输入：Questions.csv（Kaggle stackoverflow/pythonquestions）
输出：
  topic_phrases.csv      —— 每个主题最具代表性的动词短语（用来给主题命名）
  topic_assignments.csv  —— 每道题的主导主题编号和概率（供后续与采纳率合并分析）
"""

QUESTIONS_CSV = "/Users/keyue/Desktop/未命名文件夹/Questions.csv"  
OUT_DIR       = "/Users/keyue/Desktop/pythonquestions"                

N_TOPICS          = 10      
N_ROWS            = 60000  
MAX_PHRASES_PER_Q = 3       
RANDOM_STATE      = 314     


import re
import html
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


CODE_RE = re.compile(r"<code>.*?</code>", flags=re.IGNORECASE | re.DOTALL)
TAG_RE  = re.compile(r"<[^>]+>")

def clean_text(raw_html: str) -> str:
    """把 SO 的 HTML 正文变成纯文本。"""
    text = CODE_RE.sub(" ", raw_html)      
    text = TAG_RE.sub(" ", text)           
    text = html.unescape(text)             
    return " ".join(text.split())          


nlp = spacy.load("en_core_web_sm", disable=["ner"])  

MODALS = ("can", "could", "should", "would", "may", "might", "must", "do")

def extract_verb_phrases(doc, limit=MAX_PHRASES_PER_Q):
    """从一个 spaCy 文档里提取动词短语列表。"""
    phrases = []
    for token in doc:
        if token.pos_ != "VERB":
            continue
        parts = []
        if any(c.dep_ == "mark" and c.lower_ == "to" for c in token.children):
            parts.append("to")
        if any(c.dep_ == "neg" for c in token.children):
            parts.append("not")
        parts += [c.lemma_ for c in token.children
                  if c.dep_ == "aux" and c.lemma_ in MODALS]
        parts.append(token.lemma_)
        for c in token.children:
            if c.dep_ == "prt":
                parts.append(c.lemma_)
        phrases.append(" ".join(parts))
        if len(phrases) >= limit:
            break
    return phrases


print("读取 Questions.csv ...")
usecols = ["Id", "Title", "Body"]
frames = []
for chunk in pd.read_csv(QUESTIONS_CSV, usecols=usecols,
                         encoding="utf-8", encoding_errors="replace",
                         chunksize=50000):
    frames.append(chunk)
    if N_ROWS is not None and sum(len(f) for f in frames) >= N_ROWS:
        break
questions = pd.concat(frames, ignore_index=True)
if N_ROWS is not None:
    questions = questions.head(N_ROWS)
print(f"共 {len(questions):,} 道题")

print("清洗文本并提取动词短语（这一步最慢，请耐心等待）...")
questions["text"] = (questions["Title"].fillna("") + ". "
                     + questions["Body"].fillna("").map(clean_text))

docs = nlp.pipe(questions["text"], batch_size=1000, n_process=1)
questions["phrases"] = [extract_verb_phrases(doc) for doc in docs]


questions["phrase_doc"] = questions["phrases"].str.join(" | ")
usable = questions[questions["phrase_doc"] != ""].copy()
print(f"其中 {len(usable):,} 道题提取到了至少一个动词短语，进入主题模型")


print("训练 LDA 主题模型 ...")
vectorizer = CountVectorizer(
    tokenizer=lambda s: s.split(" | "),  
    token_pattern=None,                  
    min_df=20,                           
    max_df=0.5,                         
)
dtm = vectorizer.fit_transform(usable["phrase_doc"])
print(f"短语词表大小：{dtm.shape[1]:,}")

lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    random_state=RANDOM_STATE,
    learning_method="batch",
    max_iter=20,
    n_jobs=-1,
)
doc_topic = lda.fit_transform(dtm)   


vocab = vectorizer.get_feature_names_out()


rows = []
print("\n===== 各主题的代表短语（对照 A&S 五大类型命名）=====")
for k in range(N_TOPICS):
    top_idx = lda.components_[k].argsort()[::-1][:15]
    top_phrases = [vocab[i] for i in top_idx]
    print(f"\n主题 {k}: " + ", ".join(top_phrases))
    for rank, i in enumerate(top_idx, start=1):
        rows.append({"topic": k, "rank": rank, "phrase": vocab[i],
                     "weight": lda.components_[k][i]})
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/topic_phrases.csv", index=False)


usable["dominant_topic"] = doc_topic.argmax(axis=1)
usable["topic_prob"] = doc_topic.max(axis=1)
usable[["Id", "dominant_topic", "topic_prob"]].to_csv(
    f"{OUT_DIR}/topic_assignments.csv", index=False)


share = usable["dominant_topic"].value_counts(normalize=True).sort_index()
print("\n===== 各主题题目占比 =====")
for k, v in share.items():
    print(f"主题 {k}: {v:.1%}")

print(f"\n完成！结果已保存到：\n  {OUT_DIR}/topic_phrases.csv\n  {OUT_DIR}/topic_assignments.csv")
print("下一步：对照 A&S 的五大类型给每个主题命名，"
      "然后把 topic_assignments.csv 与采纳标记合并，比较各类型的回答质量。")



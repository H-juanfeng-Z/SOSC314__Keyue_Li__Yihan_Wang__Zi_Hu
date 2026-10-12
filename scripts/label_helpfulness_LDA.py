#!/usr/bin/env python3
"""
label_expanded_2000.py
======================
给 Huzi helpfulness 扩展版（expanded_v1）的 2,000 道题打 LDA 主题标签。

原理（重要）：
  - 先用和 replicate_as2013.py 完全相同的数据、参数、随机种子（314）
    重训 LDA 模型 —— 结果和上次逐位相同，这次把模型"存下来用"；
  - 模型会存档为 lda_model.joblib / vectorizer.joblib；
    重跑脚本时若发现存档，直接跳过训练（秒级）；
  - 再从原始 Questions.csv 里取出这 2,000 题的原文（不用 Huzi 包的
    body_text，保证文本处理和训练时完全一致）；
  - 用同一个向量化器 transform（套用旧模型，绝不重新 fit），
    输出每题的主导主题和置信度。

运行方法（Mac / VS Code 终端，anaconda (base) 环境）：
  python label_expanded_2000.py
产出：expanded_topic_assignments.csv（和本脚本同目录）
"""


QUESTIONS_CSV   = "/Users/keyue/Desktop/未命名文件夹/Questions.csv"
HELPFULNESS_DIR = "/Users/keyue/Desktop/sosc314-stackoverflow-response-time/data/helpfulness/releases/expanded_v1"


N_TOPICS          = 10      
N_ROWS            = 60000
MAX_PHRASES_PER_Q = 3
RANDOM_STATE      = 314


import re, html, json, os
import pandas as pd
import spacy
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_P = os.path.join(SCRIPT_DIR, "lda_model.joblib")
VEC_P   = os.path.join(SCRIPT_DIR, "vectorizer.joblib")


CODE_RE = re.compile(r"<code>.*?</code>", flags=re.IGNORECASE | re.DOTALL)
TAG_RE  = re.compile(r"<[^>]+>")

def clean_text(raw_html: str) -> str:
    text = CODE_RE.sub(" ", raw_html)
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return " ".join(text.split())

nlp = spacy.load("en_core_web_sm", disable=["ner"])
MODALS = ("can", "could", "should", "would", "may", "might", "must", "do")

def extract_verb_phrases(doc, limit=MAX_PHRASES_PER_Q):
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

def split_phrases(s):
    """向量化器的分词函数：按 " | " 把短语串切回列表。
    必须是独立命名的函数（不能用 lambda），否则模型无法存档。"""
    return s.split(" | ")

def phrases_for(texts):
    docs = nlp.pipe(texts, batch_size=1000, n_process=1)
    return [" | ".join(extract_verb_phrases(doc)) for doc in docs]


if os.path.exists(MODEL_P) and os.path.exists(VEC_P):
    print("发现模型存档，跳过训练，直接加载 ...")
    lda = joblib.load(MODEL_P)
    vectorizer = joblib.load(VEC_P)
else:
    print("读取 Questions.csv 前 60,000 行，重训 LDA ...")
    usecols = ["Id", "Title", "Body"]
    frames = []
    for chunk in pd.read_csv(QUESTIONS_CSV, usecols=usecols,
                             encoding="utf-8", encoding_errors="replace",
                             chunksize=50000):
        frames.append(chunk)
        if sum(len(f) for f in frames) >= N_ROWS:
            break
    train = pd.concat(frames, ignore_index=True).head(N_ROWS)
    train["text"] = (train["Title"].fillna("") + ". "
                     + train["Body"].fillna("").map(clean_text))
    print("  提取训练集动词短语（最慢的一步，请耐心）...")
    train["phrase_doc"] = phrases_for(train["text"])
    train = train[train["phrase_doc"] != ""].copy()
    print(f"  {len(train):,} 道题进入训练")

    vectorizer = CountVectorizer(tokenizer=split_phrases,
                                 token_pattern=None, min_df=20, max_df=0.5)
    dtm = vectorizer.fit_transform(train["phrase_doc"])
    lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=RANDOM_STATE,
                                    learning_method="batch", max_iter=20, n_jobs=-1)
    lda.fit(dtm)
    joblib.dump(lda, MODEL_P)
    joblib.dump(vectorizer, VEC_P)
    print(f"  训练完成，词表 {dtm.shape[1]:,}，模型已存档")

    # 自检：与旧结果对比（如果旧文件在的话）
    old_path = "/Users/keyue/Desktop/pythonquestions/topic_assignments.csv"
    if os.path.exists(old_path):
        old = pd.read_csv(old_path)
        new_assign = lda.transform(vectorizer.transform(train["phrase_doc"])).argmax(axis=1)
        same = (old.set_index("Id").loc[train["Id"], "dominant_topic"].values == new_assign).mean()
        print(f"  自检：与旧结果主导主题一致率 {same:.1%}（应接近 100%）")


print("\n读取 helpfulness 题目清单 ...")
with open(os.path.join(HELPFULNESS_DIR, "questions.json")) as f:
    qlist = json.load(f)
need_ids = sorted({q["question_id"] for q in qlist})
print(f"  需要标注 {len(need_ids):,} 道题")

print("从原始 Questions.csv 中查找这些题 ...")
usecols = ["Id", "Title", "Body"]
found = []
for chunk in pd.read_csv(QUESTIONS_CSV, usecols=usecols,
                         encoding="utf-8", encoding_errors="replace",
                         chunksize=50000):
    hit = chunk[chunk["Id"].isin(need_ids)]
    if len(hit):
        found.append(hit)
target = pd.concat(found, ignore_index=True).drop_duplicates("Id")
missing = set(need_ids) - set(target["Id"])
if missing:
    print(f"  注意：{len(missing)} 道题在原始 CSV 中未找到：{sorted(missing)[:5]} ...")

target["text"] = (target["Title"].fillna("") + ". "
                  + target["Body"].fillna("").map(clean_text))
print("  提取目标题动词短语 ...")
target["phrase_doc"] = phrases_for(target["text"])
labeled = target[target["phrase_doc"] != ""].copy()
no_phrase = set(target["Id"]) - set(labeled["Id"])
print(f"  {len(labeled):,} 道题成功提取短语；{len(no_phrase)} 道题无动词短语（无法打标）")

print("  用训练好的模型 transform ...")
doc_topic = lda.transform(vectorizer.transform(labeled["phrase_doc"]))
labeled["dominant_topic"] = doc_topic.argmax(axis=1)
labeled["topic_prob"] = doc_topic.max(axis=1)


out = labeled[["Id", "dominant_topic", "topic_prob"]].sort_values("Id")
out_path = os.path.join(SCRIPT_DIR, "expanded_topic_assignments.csv")
out.to_csv(out_path, index=False)
print(f"\n完成！共标注 {len(out):,} 道题，保存到：\n  {out_path}")


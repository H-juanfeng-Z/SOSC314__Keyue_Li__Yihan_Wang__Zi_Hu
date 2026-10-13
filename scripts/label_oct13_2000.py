#!/usr/bin/env python3
"""
label_oct13_2000.py
===================
给 Huzi 新包 oct13_overnight_v1 的 2,000 道题打 LDA 意图标签。

流程与 label_expanded_2000.py 完全一致（同一模型存档、同一条文本流水线）：
  - 从原始 Questions.csv 按 ID 取 title + body_html（严格口径）；
  - 复用桌面上存档的 lda_model.joblib / vectorizer.joblib（秒级加载，不重训）；
  - 找不到原文的题单独标记为 missing_source，不用近似文本顶替。

产出（和本脚本同目录）：
  as_lda_classifications_oct13.json   交给 Huzi 的分类列表（他的格式）
  oct13_topic_assignments.csv         我们自己的明细（含 topic 编号/置信度/来源标记）

运行方法（Mac / VS Code 终端，anaconda (base) 环境，本脚本放在桌面）：
  python label_oct13_2000.py
"""

import json, os
import pandas as pd
import spacy
import joblib

# ============================================================
# 第 0 步：配置区
# ============================================================
QUESTIONS_CSV   = "/Users/keyue/Desktop/未命名文件夹/Questions.csv"
INDEX_JSON      = "/Users/keyue/Desktop/question_index.json"   
MODEL_P         = "/Users/keyue/Desktop/lda_model.joblib"
VEC_P           = "/Users/keyue/Desktop/vectorizer.joblib"
CLASSIFICATION_VERSION = "as_lda_five_types_seed314_oct13"

TOPIC2TYPE = {0:'implement',1:'implement',2:'implement',3:'how_why',4:'way_of_using',
              5:'way_of_using',6:'how_why',7:'do_not_work',8:'learning',9:'way_of_using'}

# ------------------------------------------------------------
# 必须在加载 vectorizer 存档之前定义（存档里记录了它的名字）
# ------------------------------------------------------------
def split_phrases(s):
    return s.split(" | ")

# ------------------------------------------------------------
# 与 replicate_as2013.py 相同的清洗 + 动词短语提取
# ------------------------------------------------------------
import re, html
CODE_RE = re.compile(r"<code>.*?</code>", flags=re.IGNORECASE | re.DOTALL)
TAG_RE  = re.compile(r"<[^>]+>")

def clean_text(raw_html: str) -> str:
    text = CODE_RE.sub(" ", raw_html)
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return " ".join(text.split())

nlp = spacy.load("en_core_web_sm", disable=["ner"])
MODALS = ("can", "could", "should", "would", "may", "might", "must", "do")

def extract_verb_phrases(doc, limit=3):
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# 第 1 步：加载模型 + 题目清单
# ------------------------------------------------------------
print("加载模型存档 ...")
lda = joblib.load(MODEL_P)
vectorizer = joblib.load(VEC_P)

with open(INDEX_JSON) as f:
    qindex = json.load(f)
need_ids = sorted({x["question_id"] for x in qindex})
print(f"待标注 {len(need_ids):,} 道题")

# ------------------------------------------------------------
# 第 2 步：从原始 CSV 找原文
# ------------------------------------------------------------
print("从原始 Questions.csv 中查找 ...")
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
print(f"  找到 {len(target):,} 题；未找到 {len(missing)} 题（标记 missing_source）")

target["text"] = (target["Title"].fillna("") + ". "
                  + target["Body"].fillna("").map(clean_text))
print("提取动词短语 ...")
docs = nlp.pipe(target["text"], batch_size=1000, n_process=1)
target["phrase_doc"] = [" | ".join(extract_verb_phrases(doc)) for doc in docs]
labeled = target[target["phrase_doc"] != ""].copy()
no_phrase = set(target["Id"]) - set(labeled["Id"])
print(f"  {len(labeled):,} 题成功提取短语；{len(no_phrase)} 题无动词短语")

# ------------------------------------------------------------
# 第 3 步：transform 打标签
# ------------------------------------------------------------
doc_topic = lda.transform(vectorizer.transform(labeled["phrase_doc"]))
labeled["dominant_topic"] = doc_topic.argmax(axis=1)
labeled["topic_prob"] = doc_topic.max(axis=1)
labeled["as_type"] = labeled["dominant_topic"].map(TOPIC2TYPE)

# ------------------------------------------------------------
# 第 4 步：产出 1 —— 给 Huzi 的分类列表（他的格式）
# ------------------------------------------------------------
classifications = []
for _, r in labeled.iterrows():
    classifications.append({
        "question_id": int(r["Id"]),
        "classification_version": CLASSIFICATION_VERSION,
        "status": "valid",
        "categories": [r["as_type"]],
    })
for qid in sorted(missing):
    classifications.append({
        "question_id": int(qid),
        "classification_version": CLASSIFICATION_VERSION,
        "status": "missing_source",
        "categories": [],
    })
classifications.sort(key=lambda x: x["question_id"])
out1 = os.path.join(SCRIPT_DIR, "as_lda_classifications_oct13.json")
with open(out1, "w") as f:
    json.dump(classifications, f, ensure_ascii=False, indent=1)

# ------------------------------------------------------------
# 第 5 步：产出 2 —— 我们自己的明细表
# ------------------------------------------------------------
detail = labeled[["Id", "dominant_topic", "topic_prob", "as_type"]].rename(
    columns={"Id": "question_id"}).sort_values("question_id")
detail["text_source"] = "original_csv"
out2 = os.path.join(SCRIPT_DIR, "oct13_topic_assignments.csv")
detail.to_csv(out2, index=False)

print(f"\n完成！")
print(f"  交给 Huzi：{out1}（valid {len(labeled):,} 条，missing_source {len(missing)} 条）")
print(f"  自留明细：{out2}")
print(f"  无动词短语未标注：{len(no_phrase)} 题（已含在 missing 统计外的说明里）")
print("把两个文件都发给 Rico 的 AI 助手即可。")

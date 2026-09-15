# -*- coding: utf-8 -*-
import re
from bs4 import BeautifulSoup

from preprocess import uppercase_ratio

WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:[.'][A-Za-z0-9_]+)*")

# Regular expressions for identifying common error messages.
ERROR_PATTERNS = [
    re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:Error|Exception)\b"),
    re.compile(r"\btraceback\s*\(most recent call last\)", re.I),
    re.compile(r"\bstack\s*trace\b", re.I),
    re.compile(
        r"\b(errno|exit code|segmentation fault|syntax error|fatal error)\b", re.I
    ),
]


def word_count(text):
    """Count word-like tokens in a text string."""
    return len(WORD_RE.findall(text or ""))


def extract_features(title, body_html, tag_count=None):
    """
    Extract observable characteristics from one Stack Overflow question.

    The function focuses on question detail, code/technical information,
    presentation structure, and specificity. Affect-related features are
    intentionally excluded.
    """
    title = title or ""
    body_html = body_html or ""

    soup = BeautifulSoup(body_html, "html.parser")

    # Code-related features.
    code_texts = []
    for tag in soup.find_all("pre"):
        code_texts.append(tag.get_text())

    for tag in soup.find_all("code"):
        if tag.parent is not None and tag.parent.name == "pre":
            continue
        code_texts.append(tag.get_text())

    code_full_text = "\n".join(code_texts)
    code_length = word_count(code_full_text)
    has_code = int(code_length > 0)

    # Natural-language explanatory text after removing code.
    soup_no_code = BeautifulSoup(body_html, "html.parser")
    for tag in soup_no_code.find_all(["pre", "code"]):
        tag.decompose()

    explanatory_text = soup_no_code.get_text(separator=" ")
    explanatory_length = word_count(explanatory_text)

    body_text_all = soup.get_text(separator=" ")
    body_word_count = word_count(body_text_all)
    title_word_count = word_count(title)

    code_text_ratio = code_length / max(1, code_length + explanatory_length)

    # Presentation and structural features.
    paragraphs = len(soup.find_all("p"))
    list_items = soup.find_all(["ul", "ol", "li"])
    has_list = int(len(list_items) > 0)

    links = soup.find_all("a", href=True)
    url_regex_hits = re.findall(r"https?://\S+", body_text_all)
    has_url = int(len(links) > 0 or len(url_regex_hits) > 0)

    uppercase_ratio_value = uppercase_ratio(title + " " + body_text_all)

    question_marks = title.count("?") + body_text_all.count("?")

    # Specificity feature: presence of recognizable error messages.
    combined_text = title + " " + body_text_all + " " + code_full_text
    error_hits = []
    for pattern in ERROR_PATTERNS:
        error_hits.extend(pattern.findall(combined_text))

    has_error_message = int(len(error_hits) > 0)

    result = {
        "body_word_count": body_word_count,
        "title_word_count": title_word_count,
        "question_marks": question_marks,
        "explanatory_text_length": explanatory_length,
        "has_code": has_code,
        "code_length": code_length,
        "code_text_ratio": round(code_text_ratio, 3),
        "paragraphs": paragraphs,
        "has_list": has_list,
        "has_url": has_url,
        "uppercase_ratio": round(uppercase_ratio_value, 3),
        "has_error_message": has_error_message,
    }

    if tag_count is not None:
        result["tag_count"] = tag_count

    return result

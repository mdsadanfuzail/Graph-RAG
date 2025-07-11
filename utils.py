import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

def preprocess_text(text: str) -> str:
    return re.sub(r"[^\w\s.]", "", text.lower())

def f1_score(pred: str, ref: str) -> float:
    pred_tokens = set(word_tokenize(pred.lower()))
    ref_tokens = set(word_tokenize(ref.lower()))
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

def exact_match(pred: str, ref: str) -> int:
    return int(pred.strip().lower() == ref.strip().lower())

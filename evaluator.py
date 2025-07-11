from utils import exact_match, f1_score

def evaluate(predictions, references):
    em = sum(exact_match(p, r) for p, r in zip(predictions, references)) / len(references)
    f1 = sum(f1_score(p, r) for p, r in zip(predictions, references)) / len(references)
    return em, f1

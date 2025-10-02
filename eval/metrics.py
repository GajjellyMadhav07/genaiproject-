from typing import List, Tuple

import evaluate
import sacrebleu
from rouge_score import rouge_scorer


def bleu_score(references: List[str], predictions: List[str]) -> float:
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return float(bleu.score)


def rouge_l_f1(references: List[str], predictions: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(references, predictions)]
    return float(sum(scores) / max(1, len(scores)))


def precision_recall_at_k(relevant: List[List[str]], retrieved: List[List[str]], k: int) -> Tuple[float, float]:
    precisions = []
    recalls = []
    for rel, ret in zip(relevant, retrieved):
        topk = ret[:k]
        hit = len(set(rel) & set(topk))
        precisions.append(hit / max(1, len(topk)))
        recalls.append(hit / max(1, len(rel)))
    return float(sum(precisions) / max(1, len(precisions))), float(sum(recalls) / max(1, len(recalls)))



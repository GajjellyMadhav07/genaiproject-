import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any

from eval.metrics import bleu_score, rouge_l_f1, precision_recall_at_k


@dataclass
class Example:
    prompt: str
    reference_code: str
    relevant_snippets: List[str]


def run_benchmark(
    examples: List[Example],
    generator_fn,
) -> Dict[str, Any]:
    preds: List[str] = []
    refs: List[str] = []
    retrieved: List[List[str]] = []
    relevant: List[List[str]] = []
    latencies: List[float] = []

    for ex in examples:
        t0 = time.perf_counter()
        pred, retrieved_snippets = generator_fn(ex.prompt)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        preds.append(pred)
        refs.append(ex.reference_code)
        relevant.append(ex.relevant_snippets)
        retrieved.append(retrieved_snippets)

    bleu = bleu_score(refs, preds)
    rouge = rouge_l_f1(refs, preds)
    p_at_5, r_at_5 = precision_recall_at_k(relevant, retrieved, k=5)

    return {
        "bleu": bleu,
        "rougeL_f1": rouge,
        "precision@5": p_at_5,
        "recall@5": r_at_5,
        "avg_latency_ms": sum(latencies) / max(1, len(latencies)),
    }


def main() -> None:
    # Placeholder: simple echo generator; replace with actual model invocations
    def fake_generator(prompt: str):
        return "print('demo')\n", ["snippet1", "snippet2"]

    examples = [
        Example(
            prompt="Write a function to add two integers in Python",
            reference_code="def add(a,b):\n    return a+b\n",
            relevant_snippets=["addition", "math"],
        )
    ]
    report = run_benchmark(examples, fake_generator)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()



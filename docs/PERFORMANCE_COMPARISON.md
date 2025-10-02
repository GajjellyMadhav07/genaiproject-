# Performance Comparison

This document summarizes baseline vs. optimized model results.

## Scenarios
- Short algorithmic tasks (e.g., Fibonacci, sorting)
- Web backend snippets (Flask routes, SQL)
- Unit test generation

## Metrics
- BLEU, ROUGE-L F1 for code similarity
- Precision@K / Recall@K for retrieval (if used)
- Latency and throughput

## Baseline
- HF Inference API model: `bigcode/starcoder2-3b`
- Default parameters: max_new_tokens=256, temperature=0.2, top_p=0.9

## Optimized (example)
- LoRA-adapted model on domain data
- Reduced max_new_tokens where possible
- Prompt templates tuned per task

## Reporting
Use `eval/benchmark.py` to produce a JSON report, then compare runs over time.

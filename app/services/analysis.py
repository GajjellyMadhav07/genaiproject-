from typing import Dict, Any
import re

import nltk
from radon.complexity import cc_visit


def ensure_nltk() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


def tokenize_code(code: str) -> list[str]:
    ensure_nltk()
    # Naive tokenization: identifiers, keywords, symbols
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\S", code)
    return tokens


def cyclomatic_complexity(code: str) -> float:
    try:
        results = cc_visit(code)
        if not results:
            return 0.0
        return float(sum(r.complexity for r in results) / len(results))
    except Exception:
        return 0.0


def detect_patterns(code: str) -> list[str]:
    patterns: list[tuple[str, str]] = [
        (r"class\s+\w+", "class_definition"),
        (r"def\s+\w+\s*\(", "function_definition"),
        (r"for\s+.+:\s*$", "for_loop"),
        (r"while\s+.+:\s*$", "while_loop"),
        (r"try:\s*$", "try_block"),
        (r"except\s+.+:\s*$", "except_block"),
        (r"if\s+.+:\s*$", "if_statement"),
    ]
    found: list[str] = []
    for regex, name in patterns:
        if re.search(regex, code, flags=re.MULTILINE):
            found.append(name)
    return found


def analyze_code(code: str) -> Dict[str, Any]:
    tokens = tokenize_code(code)
    complexity = cyclomatic_complexity(code)
    patterns = detect_patterns(code)
    return {
        "token_count": len(tokens),
        "avg_cyclomatic_complexity": complexity,
        "patterns": patterns,
    }





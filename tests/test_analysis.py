from app.services.analysis import analyze_code, tokenize_code, detect_patterns


def test_analyze_code_basic():
    code = """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
"""
    result = analyze_code(code)
    assert result["token_count"] > 0
    assert isinstance(result["avg_cyclomatic_complexity"], float)
    pats = detect_patterns(code)
    assert "function_definition" in pats

from app.services.analysis import analyze_code


def test_analyze_code_basic():
    code = """
def fib(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n-1):
        a, b = b, a+b
    return b
"""
    result = analyze_code(code)
    assert "token_count" in result
    assert result["avg_cyclomatic_complexity"] >= 0
    assert isinstance(result["patterns"], list)





"""Question-type analysis: keywords, OCR misreads, and the shape of the options."""
import pytest

from examocr.classify import analyze_type
from examocr.parse import parse_question

CASES = [
    ("mcq", "12. The SI unit of charge is\n(1) Ampere (2) Coulomb (3) Volt (4) Ohm", False, False),
    # (a)-(d) options on two lines must NOT count as "numbered statements" (real CET page bug)
    ("mcq", "1. System in which there is no exchange of matter,\nwork or energy from surroundings is\n"
     "(a) closed (b) adiabatic\n(c) isolated (d) isothermal.", False, False),
    ("assertion_reason", "5. Given below are two statements:\nAssertion (A): x is y.\nReason (R): z.\n"
     "(1) Both A and R are true and R explains A (2) Both A and R are true but R does not explain A\n"
     "(3) A is true but R is false (4) A is false but R is true", False, False),
    # 'Assertion' misread by OCR - the options still give it away
    ("assertion_reason", "5. Assertlon (A): x is y.\nReasen (R): z.\n"
     "(1) Both A and R are true (2) A is true but R is false (3) A is false but R is true (4) none", False, False),
    ("match", "7. Match List-I with List-II\n(A) Force (I) Joule\n(B) Work (II) Newton\n"
     "(1) A-II, B-I (2) A-I, B-II (3) A-II, B-II (4) A-I, B-I", False, True),
    # no 'Column' keyword at all, options are pairings
    ("match", "7. Pick the right pairs\n(1) P-ii, Q-i, R-iii (2) P-i, Q-ii, R-iii (3) P-iii, Q-i, R-ii "
     "(4) P-ii, Q-iii, R-i", False, False),
    ("statement", "9. Consider the following statements:\n(a) X is true\n(b) Y is true\n(c) Z is true\n"
     "Which of the following statements are correct?\n(1) (a) and (b) only (2) (b) and (c) only "
     "(3) (a) and (c) only (4) (a), (b) and (c)", False, False),
    ("arrange", "14. Arrange the following in increasing order of size\nA. Na B. Mg C. Al D. Si\n"
     "(1) A, B, C, D (2) D, C, B, A (3) B, A, D, C (4) C, D, A, B", False, False),
    ("arrange", "14. Choose the sequence of events\n(1) B, A, D, C (2) A, B, C, D (3) D, C, B, A (4) C, A, B, D",
     False, False),
    ("multi_correct", "3. One or more than one options are correct. Which are vectors?\n"
     "(A) Force (B) Mass (C) Velocity (D) Time", False, False),
    ("numerical", "21. The value of x is ____ (nearest integer).", False, False),
    ("numerical", "21. Find the work done in joules.", False, False),
    ("mcq", "8. If n is an integer, the value of (-1)^(2n) is\n(1) 1 (2) -1 (3) 0 (4) n", False, False),
    ("diagram", "4. In the circuit shown, find the current.\n(1) 1 A (2) 2 A (3) 3 A (4) 4 A", True, False),
    ("other", "11. How many of the following are noble gases? He, Ne, N, Ar\n(1) 1 (2) 2 (3) 3 (4) 4",
     False, False),
]


@pytest.mark.parametrize("expected,text,diagram,table", CASES)
def test_type(expected, text, diagram, table):
    p = parse_question(text)
    tp, conf, signals = analyze_type(text, p["options"], p["statements"], diagram, table, False, "NEET")
    assert tp.value == expected, (tp, conf, signals)
    assert signals


def test_confident_on_clear_cases():
    t = CASES[1][1]
    p = parse_question(t)
    _, conf, _ = analyze_type(t, p["options"], p["statements"])
    assert conf >= 0.8

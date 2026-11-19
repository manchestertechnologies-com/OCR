from examocr.classify import classify
from examocr.parse import parse_question, split_options
from examocr.schemas import QuestionType
from examocr.segment import match_marker


def test_marker_forms():
    assert match_marker("Q1. A body", None, False) == 1
    assert match_marker("Que 12 The", 12, False) == 12
    assert match_marker("12) The", 12, False) == 12
    assert match_marker("0.7 Find", 7, False) == 7          # OCR read Q as 0
    assert match_marker("(1) 6 J", 2, False) is None         # option, not a question
    assert match_marker("45. Far ahead", 3, False) is None   # breaks the sequence
    assert match_marker("1. Physics restart", 31, True) == 1  # numbering restarts after a section


def test_options_are_exact_substrings():
    text = "Kinetic energy is\n(1) 6 J (2) 9 J (3) 12 J (4) 18 J"
    stem, opts = split_options(text)
    assert stem == "Kinetic energy is"
    assert opts == ["(1) 6 J", "(2) 9 J", "(3) 12 J", "(4) 18 J"]
    assert all(o in text for o in opts)


def test_statements_kept_apart_from_options():
    t = ("Given below are two statements:\nAssertion (A): x\nReason (R): y\n"
         "(1) Both true (2) A true (3) R true (4) none")
    p = parse_question(t)
    assert len(p["options"]) == 4
    assert p["statements"][0].startswith("Assertion")


def test_types():
    assert classify("Assertion (A): ... Reason (R): ...", 4, False, False)[0] == QuestionType.assertion_reason
    assert classify("Match Column I with Column II", 4, False, False)[0] == QuestionType.match
    assert classify("Assertlon (A): misread ... Reason (R)", 4, False, False)[0] == QuestionType.assertion_reason
    assert classify("Find the value of x", 0, False, False)[0] == QuestionType.numerical
    assert classify("Which of these has one or more correct", 4, False, False)[0] == QuestionType.multi_correct
    assert classify("Find the current", 4, True, False)[0] == QuestionType.diagram
    assert classify("anything", 4, False, True)[0] == QuestionType.paragraph

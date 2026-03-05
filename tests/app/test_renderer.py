"""
Tests for app/renderer.py — no Qt required.
"""
from __future__ import annotations

import pytest

from app.renderer import chapter_text_to_html, result_to_html, similar_to_html
from horeb.schemas import (
    BookAnalysisResult,
    OutlineSection,
    PassageAnalysisResult,
    SimilarityResult,
    SimilarOverlap,
    VerseCitation,
)


# ---------------------------------------------------------------------------
# Helpers — minimal schema objects
# ---------------------------------------------------------------------------

def _make_passage_result(**overrides) -> PassageAnalysisResult:
    defaults = dict(
        summary=["God so loved the world.", "Belief leads to eternal life."],
        key_themes=["Love", "Salvation"],
        citations=[VerseCitation(verse_reference="John 3:16", quoted_text="For God so loved")],
        low_confidence_fields=[],
    )
    defaults.update(overrides)
    return PassageAnalysisResult(**defaults)


def _make_book_result(**overrides) -> BookAnalysisResult:
    defaults = dict(
        summary=["Ruth shows loyal love.", "Redemption through Boaz."],
        key_themes=["Loyalty", "Redemption"],
        outline=[
            OutlineSection(
                title="Ruth Stays with Naomi",
                start_verse="1:1",
                end_verse="1:22",
                source_segments=[0],
                summary="Ruth refuses to leave Naomi.",
            )
        ],
        failed_segments=[],
        low_confidence_fields=[],
    )
    defaults.update(overrides)
    return BookAnalysisResult(**defaults)


def _make_similar_result(**overrides) -> SimilarityResult:
    defaults = dict(
        seed_ref="John 3:16",
        candidates=[
            SimilarOverlap(
                candidate_ref="Romans 5:8",
                verbatim_seed_quote="For God so loved",
                verbatim_candidate_quote="God commendeth his love",
                overlap_terms=["love", "God"],
                similarity_score=0.72,
            )
        ],
    )
    defaults.update(overrides)
    return SimilarityResult(**defaults)


# ---------------------------------------------------------------------------
# chapter_text_to_html
# ---------------------------------------------------------------------------

_SAMPLE_CHAPTER_TEXT = (
    "[3:16] For God so loved the world, that he gave his only begotten Son,\n"
    "[3:17] For God sent not the Son into the world to judge the world;\n"
)


def test_chapter_text_has_verse_superscripts():
    html = chapter_text_to_html(_SAMPLE_CHAPTER_TEXT, chapter=3)
    assert "<sup" in html
    assert ">16<" in html
    assert ">17<" in html


def test_chapter_text_has_chapter_header():
    html = chapter_text_to_html(_SAMPLE_CHAPTER_TEXT, chapter=3)
    assert "Chapter 3" in html


def test_chapter_text_strips_bracket_labels():
    html = chapter_text_to_html(_SAMPLE_CHAPTER_TEXT, chapter=3)
    assert "[3:16]" not in html
    assert "[3:17]" not in html


# ---------------------------------------------------------------------------
# result_to_html — PassageAnalysisResult
# ---------------------------------------------------------------------------

def test_result_to_html_passage_summary():
    result = _make_passage_result()
    html = result_to_html(result, "John 3:16-21")
    assert "God so loved the world" in html
    assert "Belief leads to eternal life" in html


def test_result_to_html_passage_themes():
    result = _make_passage_result()
    html = result_to_html(result, "John 3:16-21")
    assert "Love" in html
    assert "Salvation" in html


def test_result_to_html_passage_citations():
    result = _make_passage_result()
    html = result_to_html(result, "John 3:16-21")
    assert "John 3:16" in html
    assert "For God so loved" in html


# ---------------------------------------------------------------------------
# result_to_html — BookAnalysisResult
# ---------------------------------------------------------------------------

def test_result_to_html_book_outline():
    result = _make_book_result()
    html = result_to_html(result, "Ruth")
    assert "Ruth Stays with Naomi" in html
    assert "1:1" in html


def test_result_to_html_failed_segments_note():
    result = _make_book_result(failed_segments=[2, 5])
    html = result_to_html(result, "Ruth")
    assert "segment" in html.lower()
    assert "2" in html
    assert "5" in html


def test_result_to_html_low_confidence_note():
    result = _make_passage_result(low_confidence_fields=["citations"])
    html = result_to_html(result, "John 3:16")
    assert "citations" in html


# ---------------------------------------------------------------------------
# similar_to_html
# ---------------------------------------------------------------------------

def test_similar_to_html_ranked_list():
    result = _make_similar_result()
    html = similar_to_html(result)
    assert "Romans 5:8" in html
    assert "0.7200" in html


def test_similar_to_html_with_tags():
    candidate = SimilarOverlap(
        candidate_ref="Romans 5:8",
        verbatim_seed_quote="For God so loved",
        verbatim_candidate_quote="God commendeth his love",
        overlap_terms=["love", "God"],
        similarity_score=0.72,
        tag="shared_phrase",
        justification_terms=["love"],
    )
    result = SimilarityResult(seed_ref="John 3:16", candidates=[candidate])
    html = similar_to_html(result)
    assert "shared_phrase" in html


def test_similar_to_html_empty_candidates():
    result = SimilarityResult(seed_ref="John 3:16", candidates=[])
    html = similar_to_html(result)
    assert "No similar passages found" in html

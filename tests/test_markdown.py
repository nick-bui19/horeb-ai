"""
Unit tests for markdown.py rendering functions.

All tests are pure — no LLM, no file I/O, no Bible text retrieval.
Result objects are constructed directly from schema models.

Coverage:
- extract_sections() for each result type
- render_analysis_md() for StudyGuideResult, PassageAnalysisResult, BookAnalysisResult
- render_similar_md() for SimilarityResult with and without tags
- Edge cases: empty themes, empty citations, failed_segments, low_confidence_fields
"""
import pytest

from horeb.markdown import ResultSections, extract_sections, render_analysis_md, render_similar_md
from horeb.schemas import (
    BookAnalysisResult,
    OutlineSection,
    PassageAnalysisResult,
    SimilarityResult,
    SimilarOverlap,
    StudyGuideResult,
    VerseCitation,
    Entity,
    Question,
    QuestionType,
)


# ---------------------------------------------------------------------------
# Helpers — minimal valid result construction
# ---------------------------------------------------------------------------

def _make_passage_result(
    low_confidence: list[str] | None = None,
    citations: list[dict] | None = None,
) -> PassageAnalysisResult:
    return PassageAnalysisResult(
        summary=["First point.", "Second point.", "Third point."],
        key_themes=["faith", "love", "salvation"],
        citations=[VerseCitation(**c) for c in (citations or [])],
        low_confidence_fields=low_confidence or [],
    )


def _make_study_guide_result() -> StudyGuideResult:
    return StudyGuideResult(
        summary=["Point one.", "Point two.", "Point three."],
        key_themes=["grace", "truth"],
        questions=[
            Question(type=QuestionType.COMPREHENSION, text="What is described?", verse_reference="3:16"),
            Question(type=QuestionType.COMPREHENSION, text="Who is the subject?"),
            Question(type=QuestionType.REFLECTION, text="How does this apply?"),
            Question(type=QuestionType.REFLECTION, text="What does this mean?"),
            Question(type=QuestionType.APPLICATION, text="What will you do?"),
        ],
        low_confidence_fields=[],
    )


def _make_book_result(failed: list[int] | None = None) -> BookAnalysisResult:
    return BookAnalysisResult(
        summary=["Book point one.", "Book point two.", "Book point three."],
        key_themes=["loyalty", "redemption"],
        outline=[
            OutlineSection(
                title="From Moab to Bethlehem",
                start_verse="1:1",
                end_verse="1:22",
                source_segments=[0],
                summary="Ruth follows Naomi.",
            ),
        ],
        failed_segments=failed or [],
        low_confidence_fields=[],
    )


def _make_similarity_result(with_tags: bool = False) -> SimilarityResult:
    candidates = [
        SimilarOverlap(
            candidate_ref="John 3:17",
            verbatim_seed_quote="God so loved the world",
            verbatim_candidate_quote="not to judge the world",
            overlap_terms=["world", "save"],
            similarity_score=0.4321,
            tag="shared_phrase" if with_tags else None,
            justification_terms=["world"] if with_tags else [],
        ),
    ]
    return SimilarityResult(seed_ref="John 3:16-21", candidates=candidates)


# ---------------------------------------------------------------------------
# extract_sections() — structural correctness
# ---------------------------------------------------------------------------

def test_extract_sections_passage_result() -> None:
    result = _make_passage_result(citations=[
        {"verse_reference": "3:16", "quoted_text": "God so loved"}
    ])
    sections = extract_sections(result, "John 3:16-21")

    assert sections.title == "Analysis: John 3:16-21"
    assert len(sections.summary) == 3
    assert sections.themes == ["faith", "love", "salvation"]
    assert len(sections.citations) == 1
    assert sections.citations[0] == ("3:16", "God so loved")
    assert sections.questions == []
    assert sections.outline == []


def test_extract_sections_study_guide_result() -> None:
    result = _make_study_guide_result()
    sections = extract_sections(result, "John 3:16-21")

    assert len(sections.questions) == 5
    assert sections.questions[0] == ("comprehension", "What is described?", "3:16")
    assert sections.questions[1] == ("comprehension", "Who is the subject?", None)
    assert sections.citations == []  # StudyGuideResult has no citations field


def test_extract_sections_book_result() -> None:
    result = _make_book_result()
    sections = extract_sections(result, "Ruth")

    assert sections.title == "Analysis: Ruth"
    assert len(sections.outline) == 1
    assert sections.outline[0].title == "From Moab to Bethlehem"
    assert sections.failed_segments == []
    assert sections.citations == []
    assert sections.questions == []


def test_extract_sections_empty_themes() -> None:
    result = PassageAnalysisResult(
        summary=["A.", "B.", "C."],
        key_themes=None,
        low_confidence_fields=[],
    )
    sections = extract_sections(result, "John 1:1")
    assert sections.themes == []


def test_extract_sections_low_confidence_fields() -> None:
    result = _make_passage_result(low_confidence=["key_themes", "citations"])
    sections = extract_sections(result, "John 1:1")
    assert sections.low_confidence == ["key_themes", "citations"]


def test_extract_sections_failed_segments() -> None:
    result = _make_book_result(failed=[2, 5])
    sections = extract_sections(result, "Ruth")
    assert sections.failed_segments == [2, 5]


# ---------------------------------------------------------------------------
# render_analysis_md() — section headers present
# ---------------------------------------------------------------------------

def test_render_passage_analysis_has_summary_section() -> None:
    result = _make_passage_result()
    md = render_analysis_md(result, "John 3:16-21")

    assert "# Analysis: John 3:16-21" in md
    assert "## Summary" in md
    assert "## Key Themes" in md


def test_render_passage_analysis_citations_section() -> None:
    result = _make_passage_result(citations=[
        {"verse_reference": "3:16", "quoted_text": "God so loved"},
        {"verse_reference": "3:17", "quoted_text": None},
    ])
    md = render_analysis_md(result, "John 3:16-21")

    assert "## Citations" in md
    assert "`3:16`" in md
    assert "God so loved" in md
    assert "`3:17`" in md


def test_render_study_guide_has_questions_section() -> None:
    result = _make_study_guide_result()
    md = render_analysis_md(result, "John 3:16-21")

    assert "## Study Questions" in md
    assert "comprehension" in md
    assert "What is described?" in md
    assert "3:16" in md  # verse reference rendered


def test_render_book_result_has_outline_section() -> None:
    result = _make_book_result()
    md = render_analysis_md(result, "Ruth")

    assert "## Outline" in md
    assert "From Moab to Bethlehem" in md
    assert "1:1" in md
    assert "1:22" in md


def test_render_book_result_failed_segments_notice() -> None:
    result = _make_book_result(failed=[2, 5])
    md = render_analysis_md(result, "Ruth")

    assert "2 segment" in md
    assert "[2, 5]" in md


def test_render_no_themes_shows_placeholder() -> None:
    result = PassageAnalysisResult(
        summary=["A.", "B.", "C."],
        key_themes=None,
        low_confidence_fields=[],
    )
    md = render_analysis_md(result, "John 1:1")

    assert "not determined from passage text" in md


def test_render_low_confidence_notice_present() -> None:
    result = _make_passage_result(low_confidence=["key_themes"])
    md = render_analysis_md(result, "John 1:1")

    assert "Low confidence fields" in md
    assert "key_themes" in md


def test_render_no_low_confidence_no_notice() -> None:
    result = _make_passage_result(low_confidence=[])
    md = render_analysis_md(result, "John 1:1")

    assert "Low confidence" not in md


# ---------------------------------------------------------------------------
# render_similar_md() — section content
# ---------------------------------------------------------------------------

def test_render_similar_md_has_title() -> None:
    result = _make_similarity_result()
    md = render_similar_md(result)

    assert "# Similar Passages: John 3:16-21" in md


def test_render_similar_md_has_candidate_info() -> None:
    result = _make_similarity_result()
    md = render_similar_md(result)

    assert "John 3:17" in md
    assert "0.4321" in md
    assert "world" in md
    assert "God so loved the world" in md
    assert "not to judge the world" in md


def test_render_similar_md_with_tags() -> None:
    result = _make_similarity_result(with_tags=True)
    md = render_similar_md(result)

    assert "shared_phrase" in md
    assert "`world`" in md  # justification term rendered as inline code


def test_render_similar_md_no_tag_no_tag_line() -> None:
    result = _make_similarity_result(with_tags=False)
    md = render_similar_md(result)

    assert "**Tag:**" not in md


def test_render_similar_md_empty_candidates() -> None:
    result = SimilarityResult(seed_ref="John 1:1", candidates=[])
    md = render_similar_md(result)

    assert "No similar passages found" in md
    assert "# Similar Passages: John 1:1" in md


# ---------------------------------------------------------------------------
# render_analysis_md() — output is non-empty string
# ---------------------------------------------------------------------------

def test_render_analysis_md_returns_string() -> None:
    result = _make_passage_result()
    md = render_analysis_md(result, "John 3:16-21")
    assert isinstance(md, str)
    assert len(md) > 0


def test_render_similar_md_returns_string() -> None:
    result = _make_similarity_result()
    md = render_similar_md(result)
    assert isinstance(md, str)
    assert len(md) > 0


# ---------------------------------------------------------------------------
# render_analysis_md() — outline summary truncation
# ---------------------------------------------------------------------------

def test_outline_summary_truncated_at_max() -> None:
    long_summary = "x" * 100
    result = BookAnalysisResult(
        summary=["A.", "B.", "C."],
        key_themes=["loyalty"],
        outline=[
            OutlineSection(
                title="Section",
                start_verse="1:1",
                end_verse="1:10",
                source_segments=[0],
                summary=long_summary,
            )
        ],
        failed_segments=[],
        low_confidence_fields=[],
    )
    md = render_analysis_md(result, "Ruth")

    assert long_summary not in md
    assert "…" in md

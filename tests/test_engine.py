"""
Tests for engine.py — full pipeline integration with fixture LLM providers.

Covers test matrix items:
- 8:  Valid response parses cleanly
- 10: Citation in range passes
- 14: Citation out of range → CitationOutOfRangeError
- 15: EmptyPassageError on short passage text
- 22: analyze() routing (PASSAGE/CHAPTER/BOOK dispatch)
- 23: analyze_passage() happy path and failure modes
- 24: verify_synthesis_grounding() all failure modes
- 25: find_similar() verbatim grounding and scorer stamping
"""
import json
from unittest.mock import patch

import pytest
import pythonbible as pb

from horeb.bible_text import Segment, retrieve_passage
from horeb.engine import (
    analyze,
    analyze_passage,
    analyze_study_guide,
    extract_verse_refs,
    find_similar,
    verify_citations,
    verify_synthesis_grounding,
)
from horeb.errors import (
    AnalysisFailedError,
    CitationOutOfRangeError,
    EmptyPassageError,
    InvalidReferenceError,
)
from horeb.parallels import CandidateMatch
from horeb.schemas import (
    AnalysisResult,
    BookAnalysisResult,
    OutlineSection,
    PassageAnalysisResult,
    PassageData,
    SegmentResult,
    StudyGuideResult,
)
from tests.conftest import (
    FixtureLLMProvider,
    SequentialFixtureLLMProvider,
    load_fixture,
)


# ---------------------------------------------------------------------------
# Study guide happy path (Phase 1 — uses analyze_study_guide)
# ---------------------------------------------------------------------------

class TestStudyGuideHappyPath:
    def test_valid_fixture_returns_study_guide_result(self, valid_john_llm):
        result = analyze_study_guide("John 3:16-21", llm=valid_john_llm)
        assert isinstance(result, StudyGuideResult)

    def test_result_has_3_summary_items(self, valid_john_llm):
        result = analyze_study_guide("John 3:16-21", llm=valid_john_llm)
        assert len(result.summary) == 3

    def test_result_has_5_questions(self, valid_john_llm):
        result = analyze_study_guide("John 3:16-21", llm=valid_john_llm)
        assert len(result.questions) == 5

    def test_result_question_distribution_is_2_2_1(self, valid_john_llm):
        from collections import Counter
        from horeb.schemas import QuestionType
        result = analyze_study_guide("John 3:16-21", llm=valid_john_llm)
        counts = Counter(q.type for q in result.questions)
        assert counts[QuestionType.COMPREHENSION] == 2
        assert counts[QuestionType.REFLECTION] == 2
        assert counts[QuestionType.APPLICATION] == 1


# ---------------------------------------------------------------------------
# Invalid reference propagation
# ---------------------------------------------------------------------------

class TestInvalidReferencePropagation:
    def test_invalid_reference_raises_before_llm_call(self):
        llm = FixtureLLMProvider("should not be called")
        with pytest.raises(InvalidReferenceError):
            analyze_study_guide("NotARealBook 1:1", llm=llm)
        assert llm.call_count == 0

    def test_chapter_ref_raises_in_study_guide_pipeline(self):
        # analyze_study_guide() calls retrieve_passage() which requires a verse range.
        # Chapter-only refs are accepted by the routed analyze() (Phase 2) but not here.
        llm = FixtureLLMProvider("")
        with pytest.raises(InvalidReferenceError):
            analyze_study_guide("John 3", llm=llm)


# ---------------------------------------------------------------------------
# EmptyPassageError guard
# ---------------------------------------------------------------------------

class TestEmptyPassageGuard:
    def test_empty_passage_text_raises_before_llm_call(self):
        from horeb.schemas import PassageData
        llm = FixtureLLMProvider("should not be called")

        short_passage = PassageData(
            reference="John 3:16",
            book=43,
            start_chapter=3,
            start_verse=16,
            end_chapter=3,
            end_verse=16,
            text="",
            context_before=None,
            context_after=None,
        )
        with patch("horeb.engine.retrieve_passage", return_value=short_passage):
            with pytest.raises(EmptyPassageError):
                analyze_study_guide("John 3:16", llm=llm)

        assert llm.call_count == 0


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------

class TestCitationVerification:
    def test_in_range_citation_passes(self, valid_john_llm):
        result = analyze_study_guide("John 3:16-21", llm=valid_john_llm)
        assert isinstance(result, StudyGuideResult)

    def test_out_of_range_citation_raises(self, out_of_range_citation_llm):
        # out_of_range_citation.json cites John 3:22 which is outside 3:16-21
        with pytest.raises(CitationOutOfRangeError):
            analyze_study_guide("John 3:16-21", llm=out_of_range_citation_llm)

    def test_extract_verse_refs_from_questions(self, valid_john_llm):
        result = analyze_study_guide("John 3:16-21", llm=valid_john_llm)
        refs = extract_verse_refs(result)
        assert len(refs) > 0
        for ref in refs:
            assert ":" in ref

    def test_extract_verse_refs_skips_null_references(self):
        from horeb.schemas import Question, QuestionType
        result = AnalysisResult(
            summary=["A.", "B.", "C."],
            questions=[
                Question(type=QuestionType.COMPREHENSION, text="Q1?", verse_reference="3:16"),
                Question(type=QuestionType.COMPREHENSION, text="Q2?", verse_reference=None),
                Question(type=QuestionType.REFLECTION, text="Q3?", verse_reference="3:19"),
                Question(type=QuestionType.REFLECTION, text="Q4?", verse_reference="3:21"),
                Question(type=QuestionType.APPLICATION, text="Q5?", verse_reference=None),
            ],
        )
        refs = extract_verse_refs(result)
        assert refs == ["3:16", "3:19", "3:21"]


# ---------------------------------------------------------------------------
# AnalysisFailedError propagation
# ---------------------------------------------------------------------------

class TestAnalysisFailedPropagation:
    def test_analysis_failed_raises_with_wrong_distribution_and_bad_retry(self):
        wrong = load_fixture("wrong_question_distribution.json")
        # analyze_study_guide() calls llm.complete() once, then repair_and_validate
        # retries once → 2 total LLM calls, both return wrong distribution
        llm = SequentialFixtureLLMProvider([wrong, wrong])
        with pytest.raises(AnalysisFailedError):
            analyze_study_guide("John 3:16-21", llm=llm)


# ---------------------------------------------------------------------------
# Helpers shared by new test classes
# ---------------------------------------------------------------------------

def _make_seg_result(idx: int) -> SegmentResult:
    return SegmentResult(
        segment_index=idx,
        outline_label="Section label here",
        summary=["A.", "B.", "C."],
    )


def _make_segment(idx: int, start_ch: int, end_ch: int) -> Segment:
    return Segment(
        book=pb.Book.RUTH,
        segment_index=idx,
        start_chapter=start_ch,
        start_verse=1,
        end_chapter=end_ch,
        end_verse=22,
        verse_count=22,
        text="[1:1] In the days when the judges ruled.",
        reference=f"Ruth {start_ch}:1-{end_ch}:22",
    )


_FAKE_SEED_TEXT = "[3:16] God so loved the world that he gave his Son."
_FAKE_CANDIDATE_TEXT = "[1:12] As many as believed on his name received him."


def _make_seed_passage() -> PassageData:
    return PassageData(
        reference="John 3:16-21",
        book=43,
        start_chapter=3,
        start_verse=16,
        end_chapter=3,
        end_verse=21,
        text=_FAKE_SEED_TEXT,
        context_before=None,
        context_after=None,
    )


def _make_candidate_match() -> CandidateMatch:
    return CandidateMatch(
        reference="John 1:12",
        text=_FAKE_CANDIDATE_TEXT,
        similarity_score=0.75,
        overlap_terms=["believed", "name"],
    )


# ---------------------------------------------------------------------------
# analyze() routing dispatch
# ---------------------------------------------------------------------------

class TestAnalyzeRouting:
    def test_passage_ref_routes_to_analyze_passage(self):
        mock_result = PassageAnalysisResult(summary=["A.", "B.", "C."], citations=[])
        with patch("horeb.engine.analyze_passage", return_value=mock_result) as mock_ap:
            with patch("horeb.engine.retrieve_passage"):
                result = analyze("John 3:16-21", llm=FixtureLLMProvider(""))
        assert isinstance(result, PassageAnalysisResult)
        assert mock_ap.called

    def test_chapter_ref_routes_to_analyze_passage(self):
        mock_result = PassageAnalysisResult(summary=["A.", "B.", "C."], citations=[])
        with patch("horeb.engine.analyze_passage", return_value=mock_result) as mock_ap:
            with patch("horeb.engine.retrieve_chapter"):
                result = analyze("John 3", llm=FixtureLLMProvider(""))
        assert isinstance(result, PassageAnalysisResult)
        assert mock_ap.called

    def test_book_ref_routes_to_analyze_book(self):
        mock_result = BookAnalysisResult(
            summary=["A.", "B.", "C."], outline=[], failed_segments=[]
        )
        with patch("horeb.engine.analyze_book", return_value=mock_result) as mock_ab:
            result = analyze("Ruth", llm=FixtureLLMProvider(""))
        assert isinstance(result, BookAnalysisResult)
        assert mock_ab.called


# ---------------------------------------------------------------------------
# analyze_passage() pipeline
# ---------------------------------------------------------------------------

class TestAnalyzePassage:
    def test_valid_fixture_returns_passage_result(self):
        llm = FixtureLLMProvider(load_fixture("passage_john_3_16_valid.json"))
        result = analyze_passage(retrieve_passage("John 3:16-21"), llm)
        assert isinstance(result, PassageAnalysisResult)

    def test_result_has_3_summary_items(self):
        llm = FixtureLLMProvider(load_fixture("passage_john_3_16_valid.json"))
        result = analyze_passage(retrieve_passage("John 3:16-21"), llm)
        assert len(result.summary) == 3

    def test_empty_passage_raises_before_llm_call(self):
        short_passage = PassageData(
            reference="John 3:16",
            book=43,
            start_chapter=3,
            start_verse=16,
            end_chapter=3,
            end_verse=16,
            text="",
            context_before=None,
            context_after=None,
        )
        llm = FixtureLLMProvider("should not be called")
        with pytest.raises(EmptyPassageError):
            analyze_passage(short_passage, llm)
        assert llm.call_count == 0

    def test_out_of_range_citation_raises(self):
        llm = FixtureLLMProvider(load_fixture("passage_john_3_16_out_of_range.json"))
        with pytest.raises(CitationOutOfRangeError):
            analyze_passage(retrieve_passage("John 3:16-21"), llm)


# ---------------------------------------------------------------------------
# verify_synthesis_grounding() failure modes
# ---------------------------------------------------------------------------

class TestVerifySynthesisGrounding:
    def _valid_book_result(self) -> BookAnalysisResult:
        return BookAnalysisResult(
            summary=["A.", "B.", "C."],
            outline=[
                OutlineSection(
                    title="Opening narrative",
                    start_verse="1:1",
                    end_verse="1:22",
                    source_segments=[0],
                )
            ],
            failed_segments=[],
        )

    def test_valid_result_passes(self):
        verify_synthesis_grounding(
            self._valid_book_result(),
            [_make_seg_result(0)],
            [_make_segment(0, 1, 1)],
        )  # must not raise

    def test_empty_source_segments_raises(self):
        book_result = BookAnalysisResult(
            summary=["A.", "B.", "C."],
            outline=[
                OutlineSection(
                    title="Empty section",
                    start_verse="1:1",
                    end_verse="1:5",
                    source_segments=[],
                )
            ],
        )
        with pytest.raises(CitationOutOfRangeError, match="empty source_segments"):
            verify_synthesis_grounding(book_result, [_make_seg_result(0)])

    def test_nonexistent_segment_index_raises(self):
        book_result = BookAnalysisResult(
            summary=["A.", "B.", "C."],
            outline=[
                OutlineSection(
                    title="Bad section",
                    start_verse="1:1",
                    end_verse="1:22",
                    source_segments=[99],
                )
            ],
        )
        with pytest.raises(CitationOutOfRangeError, match="does not exist"):
            verify_synthesis_grounding(book_result, [_make_seg_result(0)])

    def test_malformed_anchor_raises(self):
        book_result = BookAnalysisResult(
            summary=["A.", "B.", "C."],
            outline=[
                OutlineSection(
                    title="Malformed anchor",
                    start_verse="abc",
                    end_verse="1:22",
                    source_segments=[0],
                )
            ],
        )
        with pytest.raises(CitationOutOfRangeError, match="not a valid chapter:verse"):
            verify_synthesis_grounding(
                book_result, [_make_seg_result(0)], [_make_segment(0, 1, 1)]
            )

    def test_anchor_chapter_outside_segment_range_raises(self):
        book_result = BookAnalysisResult(
            summary=["A.", "B.", "C."],
            outline=[
                OutlineSection(
                    title="Wrong chapter",
                    start_verse="5:1",
                    end_verse="5:22",
                    source_segments=[0],
                )
            ],
        )
        with pytest.raises(CitationOutOfRangeError, match="outside the chapter range"):
            verify_synthesis_grounding(
                book_result, [_make_seg_result(0)], [_make_segment(0, 1, 1)]
            )


# ---------------------------------------------------------------------------
# find_similar() verbatim grounding and scorer stamping
# ---------------------------------------------------------------------------

class TestFindSimilar:
    def test_scorer_data_stamped_over_llm_fields(self):
        """Scorer overlap_terms and similarity_score overwrite LLM-provided values."""
        llm_response = json.dumps({
            "seed_ref": "John 3:16-21",
            "candidates": [{
                "candidate_ref": "John 1:12",
                "verbatim_seed_quote": "God so loved the world",
                "verbatim_candidate_quote": "believed on his name received",
                "overlap_terms": ["fabricated"],
                "similarity_score": 0.01,
            }],
        })
        llm = FixtureLLMProvider(llm_response)
        with patch("horeb.engine.retrieve_passage", return_value=_make_seed_passage()):
            with patch("horeb.engine.score_similarity", return_value=[_make_candidate_match()]):
                result = find_similar("John 3:16-21", llm=llm)
        assert result.candidates[0].similarity_score == 0.75
        assert result.candidates[0].overlap_terms == ["believed", "name"]

    def test_invented_reference_raises(self):
        llm_response = json.dumps({
            "seed_ref": "John 3:16-21",
            "candidates": [{
                "candidate_ref": "John 5:24",
                "verbatim_seed_quote": "God so loved the world",
                "verbatim_candidate_quote": "some text",
            }],
        })
        with patch("horeb.engine.retrieve_passage", return_value=_make_seed_passage()):
            with patch("horeb.engine.score_similarity", return_value=[_make_candidate_match()]):
                with pytest.raises(CitationOutOfRangeError):
                    find_similar("John 3:16-21", llm=FixtureLLMProvider(llm_response))

    def test_bad_seed_quote_raises(self):
        llm_response = json.dumps({
            "seed_ref": "John 3:16-21",
            "candidates": [{
                "candidate_ref": "John 1:12",
                "verbatim_seed_quote": "completely fabricated text not in seed",
                "verbatim_candidate_quote": "believed on his name received",
            }],
        })
        with patch("horeb.engine.retrieve_passage", return_value=_make_seed_passage()):
            with patch("horeb.engine.score_similarity", return_value=[_make_candidate_match()]):
                with pytest.raises(CitationOutOfRangeError):
                    find_similar("John 3:16-21", llm=FixtureLLMProvider(llm_response))

    def test_bad_candidate_quote_raises(self):
        llm_response = json.dumps({
            "seed_ref": "John 3:16-21",
            "candidates": [{
                "candidate_ref": "John 1:12",
                "verbatim_seed_quote": "God so loved the world",
                "verbatim_candidate_quote": "completely fabricated text not in candidate",
            }],
        })
        with patch("horeb.engine.retrieve_passage", return_value=_make_seed_passage()):
            with patch("horeb.engine.score_similarity", return_value=[_make_candidate_match()]):
                with pytest.raises(CitationOutOfRangeError):
                    find_similar("John 3:16-21", llm=FixtureLLMProvider(llm_response))

    def test_empty_candidates_returns_empty_result_without_llm_call(self):
        llm = FixtureLLMProvider("should not be called")
        with patch("horeb.engine.retrieve_passage", return_value=_make_seed_passage()):
            with patch("horeb.engine.score_similarity", return_value=[]):
                result = find_similar("John 3:16-21", llm=llm)
        assert result.seed_ref == "John 3:16-21"
        assert result.candidates == []
        assert llm.call_count == 0

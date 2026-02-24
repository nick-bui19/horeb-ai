from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Retrieval dataclass (not a Pydantic model — retrieval concern, not validation)
# ---------------------------------------------------------------------------

@dataclass
class PassageData:
    reference: str
    book: int               # pythonbible Book enum integer value
    start_chapter: int
    start_verse: int
    end_chapter: int
    end_verse: int
    text: str               # passage text with [chapter:verse] labels
    context_before: str | None  # up to CONTEXT_VERSES_BEFORE preceding verses
    context_after: str | None   # up to CONTEXT_VERSES_AFTER following verses


# ---------------------------------------------------------------------------
# Shared base model
# ---------------------------------------------------------------------------

class GroundedBase(BaseModel):
    """
    Base for all LLM output schemas.

    Enforces: exactly 3 summary items, optional themes, low_confidence_fields list.
    All subclasses inherit the summary length validator automatically.
    """
    summary: list[str]
    key_themes: list[str] | None = None
    low_confidence_fields: list[str] = []

    @model_validator(mode="after")
    def validate_summary_length(self) -> "GroundedBase":
        if len(self.summary) != 3:
            raise ValueError(
                f"summary must have exactly 3 items, got {len(self.summary)}"
            )
        return self


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class VerseCitation(BaseModel):
    """A single verse-level citation grounded in retrieved text."""
    verse_reference: str        # "chapter:verse" format, e.g. "3:16"
    quoted_text: str | None = None   # verbatim snippet from passage text


class OutlineSection(BaseModel):
    """One section of a book outline, grounded in validated segment outputs."""
    title: str                          # short section label (≤8 words)
    start_verse: str                    # "chapter:verse" anchor for section start
    end_verse: str                      # "chapter:verse" anchor for section end
    source_segments: list[int]          # indices into the segment list (grounding)
    summary: str | None = None          # optional one-sentence description


# ---------------------------------------------------------------------------
# Passage / chapter analysis result (Phase 2 analyze command — passage/chapter)
# ---------------------------------------------------------------------------

class PassageAnalysisResult(GroundedBase):
    """
    Output schema for passage-level and chapter-level analyze.
    No questions — output is: 3-bullet summary + themes + verse citations.
    """
    citations: list[VerseCitation] = []


# ---------------------------------------------------------------------------
# Per-segment result (intermediate — book pipeline stage 1)
# ---------------------------------------------------------------------------

class SegmentResult(GroundedBase):
    """
    Output schema for one book segment (typically one chapter).
    Used as input to the synthesis stage — never shown directly to the user.

    Output budget validators enforce synthesis token efficiency:
    - outline_label: ≤8 words
    - key_themes:    ≤3 items
    - citations:     ≤5 items
    """
    segment_index: int
    outline_label: str              # short section label for this segment
    citations: list[VerseCitation] = []
    source_segments: list[int] = []  # populated at synthesis stage

    @model_validator(mode="after")
    def validate_outline_label_length(self) -> "SegmentResult":
        word_count = len(self.outline_label.split())
        if word_count > 8:
            raise ValueError(
                f"outline_label must be ≤8 words, got {word_count}: {self.outline_label!r}"
            )
        return self

    @model_validator(mode="after")
    def validate_themes_count(self) -> "SegmentResult":
        if self.key_themes is not None and len(self.key_themes) > 3:
            raise ValueError(
                f"key_themes must have ≤3 items per segment, got {len(self.key_themes)}"
            )
        return self

    @model_validator(mode="after")
    def validate_citations_count(self) -> "SegmentResult":
        if len(self.citations) > 5:
            raise ValueError(
                f"citations must have ≤5 items per segment, got {len(self.citations)}"
            )
        return self


@dataclass
class SegmentFailure:
    """Represents a segment that exhausted all repair/retry attempts."""
    segment_index: int
    chapter_start: int
    chapter_end: int
    error: str


# ---------------------------------------------------------------------------
# Book analysis result (Phase 2 analyze command — whole book, synthesis output)
# ---------------------------------------------------------------------------

class BookAnalysisResult(GroundedBase):
    """
    Output schema for whole-book analyze (synthesis stage output).
    summary: 3-bullet book summary.
    outline: grounded section list with source_segments provenance.
    failed_segments: indices of segments that could not be analyzed.
    """
    outline: list[OutlineSection] = []
    failed_segments: list[int] = []


# ---------------------------------------------------------------------------
# Similarity result (Phase 2 find_similar command)
# ---------------------------------------------------------------------------

class SimilarOverlap(BaseModel):
    """
    One candidate similar passage.
    verbatim quotes are post-validated against locally retrieved text —
    the model cannot invent a parallel without the check catching it.
    """
    candidate_ref: str
    verbatim_seed_quote: str        # must appear verbatim in seed passage text
    verbatim_candidate_quote: str   # must appear verbatim in candidate passage text
    overlap_terms: list[str]        # matched tokens/phrases from TF-IDF scorer
    similarity_score: float         # from deterministic scorer, not LLM


class SimilarityResult(BaseModel):
    seed_ref: str
    candidates: list[SimilarOverlap]


# ---------------------------------------------------------------------------
# Phase 1 study guide result (backward compat — kept for StudyGuide command)
# ---------------------------------------------------------------------------

class QuestionType(str, Enum):
    COMPREHENSION = "comprehension"
    REFLECTION = "reflection"
    APPLICATION = "application"


class Entity(BaseModel):
    name: str
    type: str
    verse_reference: str | None = None
    description: str | None = None


class Question(BaseModel):
    type: QuestionType
    text: str
    verse_reference: str | None = None


class StudyGuideResult(GroundedBase):
    """
    Phase 1 study guide output: summary + themes + entities + 5 questions.
    Kept for backward compatibility with existing fixtures and tests.
    Previously named AnalysisResult.
    """
    named_entities: list[Entity] | None = None
    questions: list[Question]

    @model_validator(mode="after")
    def validate_question_distribution(self) -> "StudyGuideResult":
        counts = Counter(q.type for q in self.questions)
        expected = {
            QuestionType.COMPREHENSION: 2,
            QuestionType.REFLECTION: 2,
            QuestionType.APPLICATION: 1,
        }
        for qtype, expected_count in expected.items():
            if counts[qtype] != expected_count:
                raise ValueError(
                    f"questions must have exactly {expected_count} "
                    f"{qtype.value} questions, got {counts.get(qtype, 0)}"
                )
        return self


# Alias for backward compatibility — existing code that imports AnalysisResult continues to work.
AnalysisResult = StudyGuideResult

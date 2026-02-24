from enum import Enum
from typing import TYPE_CHECKING

import pythonbible as pb

from horeb.bible_text import retrieve_passage
from horeb.errors import (
    AnalysisFailedError,
    CitationOutOfRangeError,
    EmptyPassageError,
    InvalidReferenceError,
)
from horeb.prompts import SYSTEM_PROMPT, build_user_prompt
from horeb.repair import repair_and_validate
from horeb.schemas import AnalysisResult, PassageData, StudyGuideResult

if TYPE_CHECKING:
    from horeb.llm import LLMProvider

MIN_PASSAGE_CHARS: int = 20


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------

class CitationMode(str, Enum):
    """Controls how verify_citations validates cited references."""
    SINGLE_VERSE = "single_verse"   # passage/chapter analyze: "chapter:verse" against passage range
    SYNTHESIS = "synthesis"         # book synthesis: check against union of segment citations


def verify_citations(
    result: AnalysisResult,
    passage: PassageData,
    mode: CitationMode = CitationMode.SINGLE_VERSE,
    valid_refs: set[str] | None = None,
) -> None:
    """
    Verify every cited verse reference in result falls within the valid range.

    Args:
        result:     Any schema object with extractable verse references.
        passage:    The source PassageData (used for SINGLE_VERSE mode).
        mode:       CitationMode controlling validation strategy.
        valid_refs: For SYNTHESIS mode — the union of all segment-level citation
                    strings. Citations not in this set are out-of-scope.

    Raises:
        CitationOutOfRangeError: if any citation is outside the valid range
            or cannot be parsed.
    """
    cited_refs = extract_verse_refs(result)

    if mode == CitationMode.SYNTHESIS:
        if valid_refs is None:
            raise ValueError("valid_refs must be provided for SYNTHESIS mode")
        for ref_str in cited_refs:
            if ref_str not in valid_refs:
                raise CitationOutOfRangeError(
                    f"Synthesis cited {ref_str!r} which was not present in any "
                    f"validated segment output."
                )
        return

    # SINGLE_VERSE mode: check against passage start/end
    passage_start = (passage.start_chapter, passage.start_verse)
    passage_end = (passage.end_chapter, passage.end_verse)
    for ref_str in cited_refs:
        _check_single_verse_citation(ref_str, passage, passage_start, passage_end)


def extract_verse_refs(result: AnalysisResult) -> list[str]:
    """
    Extract all verse_reference strings from any result schema.

    Handles StudyGuideResult (questions + named_entities),
    PassageAnalysisResult and SegmentResult (citations list),
    and BookAnalysisResult (outline sections with start/end verse anchors).
    """
    refs: list[str] = []

    # StudyGuideResult — questions
    if hasattr(result, "questions") and result.questions:
        for q in result.questions:
            if q.verse_reference is not None:
                refs.append(q.verse_reference)

    # StudyGuideResult — named_entities
    if hasattr(result, "named_entities") and result.named_entities:
        for entity in result.named_entities:
            if entity.verse_reference is not None:
                refs.append(entity.verse_reference)

    # PassageAnalysisResult / SegmentResult — citations list
    if hasattr(result, "citations") and result.citations:
        for citation in result.citations:
            if citation.verse_reference:
                refs.append(citation.verse_reference)

    # BookAnalysisResult — outline section anchors
    if hasattr(result, "outline") and result.outline:
        for section in result.outline:
            if section.start_verse:
                refs.append(section.start_verse)
            if section.end_verse:
                refs.append(section.end_verse)

    return refs


def _check_single_verse_citation(
    ref_str: str,
    passage: PassageData,
    passage_start: tuple[int, int],
    passage_end: tuple[int, int],
) -> None:
    """
    Parse a single citation string and verify it is within the passage range.
    Raises CitationOutOfRangeError if out of range or unparseable.
    """
    book_name = pb.Book(passage.book).name.replace("_", " ").title()

    # Handle short "chapter:verse" format (e.g. "3:16")
    if ref_str.count(":") == 1 and not ref_str[0].isalpha():
        full_ref = f"{book_name} {ref_str}"
    else:
        full_ref = ref_str

    try:
        normalized = pb.get_references(full_ref)
        if not normalized:
            raise CitationOutOfRangeError(
                f"Cited reference {ref_str!r} could not be parsed"
            )
        cited = normalized[0]
    except CitationOutOfRangeError:
        raise
    except Exception as exc:
        raise CitationOutOfRangeError(
            f"Cited reference {ref_str!r} could not be parsed"
        ) from exc

    if cited.book.value != passage.book:
        raise CitationOutOfRangeError(
            f"Cited verse {ref_str!r} is in a different book than the passage"
        )

    cited_start = (cited.start_chapter, cited.start_verse or 0)

    if cited_start < passage_start or cited_start > passage_end:
        raise CitationOutOfRangeError(
            f"Cited verse {ref_str!r} ({cited_start}) is outside the "
            f"passage range {passage_start}–{passage_end}"
        )


# ---------------------------------------------------------------------------
# Phase 1 analyze pipeline (study guide — passage-level only)
# ---------------------------------------------------------------------------

def analyze(reference: str, llm: "LLMProvider | None" = None) -> StudyGuideResult:
    """
    Full study guide analysis pipeline: retrieve → validate → prompt → LLM → repair → verify.

    Args:
        reference: Bible passage reference string (e.g. "John 3:16-21").
        llm:       LLMProvider to use. Defaults to ClaudeProvider() if None.

    Returns:
        Validated StudyGuideResult with citations verified against the passage.

    Raises:
        InvalidReferenceError:    malformed reference or passage exceeds MAX_PASSAGE_VERSES.
        EmptyPassageError:        retrieved passage text is too short to analyse.
        CitationOutOfRangeError:  LLM cited a verse outside the retrieved passage.
        AnalysisFailedError:      LLM output failed all repair/retry attempts.
    """
    if llm is None:
        from horeb.llm import ClaudeProvider
        llm = ClaudeProvider()

    passage = retrieve_passage(reference)  # raises InvalidReferenceError

    if len(passage.text.strip()) < MIN_PASSAGE_CHARS:
        raise EmptyPassageError(
            f"Retrieved text for {reference!r} is too short "
            f"({len(passage.text)} chars). Check the reference."
        )

    user_prompt = build_user_prompt(passage)
    raw_response = llm.complete(system=SYSTEM_PROMPT, prompt=user_prompt, schema=StudyGuideResult)

    result = repair_and_validate(
        raw=raw_response,
        schema=StudyGuideResult,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    verify_citations(result, passage)

    return result

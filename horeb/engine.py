from __future__ import annotations

import sys
from enum import Enum
from typing import TYPE_CHECKING

import pythonbible as pb

from horeb.bible_text import (
    Granularity,
    MAX_BOOK_LLM_CALLS,
    detect_granularity,
    retrieve_chapter,
    retrieve_passage,
    segment_book,
)
from horeb.errors import (
    AnalysisFailedError,
    CitationOutOfRangeError,
    EmptyPassageError,
    InvalidReferenceError,
)
from horeb.prompts import (
    SYSTEM_PROMPT,
    build_passage_system_prompt,
    build_passage_user_prompt,
    build_segment_system_prompt,
    build_segment_user_prompt,
    build_synthesis_system_prompt,
    build_synthesis_user_prompt,
    build_user_prompt,
)
from horeb.repair import repair_and_validate
from horeb.schemas import (
    AnalysisResult,
    BookAnalysisResult,
    PassageAnalysisResult,
    PassageData,
    SegmentFailure,
    SegmentResult,
    StudyGuideResult,
)

if TYPE_CHECKING:
    from horeb.llm import LLMProvider

MIN_PASSAGE_CHARS: int = 20
SEGMENT_FAILURE_THRESHOLD: float = 0.3   # abort book if >30% of segments fail


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------

class CitationMode(str, Enum):
    """Controls how verify_citations validates cited references."""
    SINGLE_VERSE = "single_verse"   # passage/chapter: "chapter:verse" against passage range
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
        valid_refs: For SYNTHESIS mode — union of all segment-level citation
                    strings. Citations not in this set are out-of-scope.

    Raises:
        CitationOutOfRangeError: if any citation is outside the valid range.
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
    PassageAnalysisResult / SegmentResult (citations list),
    and BookAnalysisResult (outline section anchors).
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
# Synthesis grounding verifier
# ---------------------------------------------------------------------------

def verify_synthesis_grounding(
    book_result: BookAnalysisResult,
    segment_results: list[SegmentResult],
) -> None:
    """
    Verify that every outline section in the synthesis output is grounded in
    at least one validated segment result.

    Raises:
        CitationOutOfRangeError: if any outline section has empty source_segments,
            or references a segment index that does not exist.
    """
    valid_indices = {s.segment_index for s in segment_results}

    for i, section in enumerate(book_result.outline):
        if not section.source_segments:
            raise CitationOutOfRangeError(
                f"Outline section {i} ({section.title!r}) has empty source_segments. "
                f"Every outline section must declare its source segment indices."
            )
        for seg_idx in section.source_segments:
            if seg_idx not in valid_indices:
                raise CitationOutOfRangeError(
                    f"Outline section {i} ({section.title!r}) references segment index "
                    f"{seg_idx} which does not exist in the validated segment outputs. "
                    f"Valid indices: {sorted(valid_indices)}"
                )


# ---------------------------------------------------------------------------
# Book pipeline — two-stage analyze_book()
# ---------------------------------------------------------------------------

def analyze_book(book_name: str, llm: "LLMProvider") -> BookAnalysisResult:
    """
    Two-stage book analysis pipeline.

    Stage 1: Deterministically segment the book. For each segment, retrieve
             text and call the LLM to produce a SegmentResult. Collect
             SegmentResult | SegmentFailure outcomes.

    Stage 2: If segment failure rate is below SEGMENT_FAILURE_THRESHOLD,
             build a synthesis prompt from validated SegmentResults (with
             explicit gap markers for failures) and call the LLM to produce
             a BookAnalysisResult.

    Raises:
        InvalidReferenceError: if book name is unrecognised or produces too
            many segments (> MAX_BOOK_SEGMENTS).
        AnalysisFailedError:   if too many segments fail or synthesis fails.
    """
    # Resolve book
    try:
        refs = pb.get_references(book_name)
        if not refs:
            raise InvalidReferenceError(f"Could not find book: {book_name!r}")
        book = refs[0].book
    except InvalidReferenceError:
        raise
    except Exception as exc:
        raise InvalidReferenceError(f"Could not parse book name: {book_name!r}") from exc

    # Segment — raises InvalidReferenceError if > MAX_BOOK_SEGMENTS
    segments = segment_book(book)
    total = len(segments)

    print(
        f"[INFO] Analyzing {book.name.title()}: {total} segments, "
        f"up to {min(total * 2, MAX_BOOK_LLM_CALLS)} LLM calls.",
        file=sys.stderr,
    )

    seg_system_prompt = build_segment_system_prompt()
    segment_results: list[SegmentResult] = []
    segment_failures: list[SegmentFailure] = []
    total_llm_calls = 0

    # Stage 1: per-segment analysis
    # Segments are independent — this loop is a candidate for concurrent dispatch
    # in a future phase. See LLMProvider.complete() signature for the async path.
    for seg in segments:
        if total_llm_calls >= MAX_BOOK_LLM_CALLS:
            print(
                f"[WARN] Reached MAX_BOOK_LLM_CALLS={MAX_BOOK_LLM_CALLS}. "
                f"Stopping segment processing.",
                file=sys.stderr,
            )
            # Record remaining segments as failures
            for remaining in segments[seg.segment_index:]:
                segment_failures.append(SegmentFailure(
                    segment_index=remaining.segment_index,
                    chapter_start=remaining.start_chapter,
                    chapter_end=remaining.end_chapter,
                    error="MAX_BOOK_LLM_CALLS ceiling reached",
                ))
            break

        if len(seg.text.strip()) < MIN_PASSAGE_CHARS:
            segment_failures.append(SegmentFailure(
                segment_index=seg.segment_index,
                chapter_start=seg.start_chapter,
                chapter_end=seg.end_chapter,
                error="Segment text too short",
            ))
            continue

        user_prompt = build_segment_user_prompt(seg.text, seg.reference, seg.segment_index)

        try:
            raw = llm.complete(
                system=seg_system_prompt,
                prompt=user_prompt,
                schema=SegmentResult,
            )
            total_llm_calls += 1

            seg_result = repair_and_validate(
                raw=raw,
                schema=SegmentResult,
                llm=llm,
                system_prompt=seg_system_prompt,
                user_prompt=user_prompt,
            )
            # repair_and_validate may have made one retry call
            total_llm_calls += 1 if "[WARN]" in (raw or "") else 0

            # Stamp the segment index (model may return 0 for all segments)
            seg_result = seg_result.model_copy(update={"segment_index": seg.segment_index})
            segment_results.append(seg_result)

        except (AnalysisFailedError, Exception) as exc:
            segment_failures.append(SegmentFailure(
                segment_index=seg.segment_index,
                chapter_start=seg.start_chapter,
                chapter_end=seg.end_chapter,
                error=str(exc),
            ))

    # Check failure threshold
    failure_rate = len(segment_failures) / total if total > 0 else 0.0
    if failure_rate > SEGMENT_FAILURE_THRESHOLD:
        raise AnalysisFailedError(
            f"Too many segment failures: {len(segment_failures)}/{total} "
            f"({failure_rate:.0%} > {SEGMENT_FAILURE_THRESHOLD:.0%} threshold). "
            f"Check the book reference or reduce scope."
        )

    if segment_failures:
        failed_indices = [f.segment_index for f in segment_failures]
        print(
            f"[WARN] {len(segment_failures)} segment(s) failed: {failed_indices}. "
            f"Synthesis will include gap markers.",
            file=sys.stderr,
        )

    # Stage 2: synthesis
    syn_system_prompt = build_synthesis_system_prompt()
    syn_user_prompt = build_synthesis_user_prompt(segment_results, segment_failures)

    raw_synthesis = llm.complete(
        system=syn_system_prompt,
        prompt=syn_user_prompt,
        schema=BookAnalysisResult,
        max_tokens=4096,   # synthesis output is larger than per-segment output
    )
    total_llm_calls += 1

    book_result = repair_and_validate(
        raw=raw_synthesis,
        schema=BookAnalysisResult,
        llm=llm,
        system_prompt=syn_system_prompt,
        user_prompt=syn_user_prompt,
        max_tokens=4096,
    )

    # Stamp failed segment indices into result
    if segment_failures:
        book_result = book_result.model_copy(
            update={"failed_segments": [f.segment_index for f in segment_failures]}
        )

    # Verify synthesis grounding
    verify_synthesis_grounding(book_result, segment_results)

    return book_result


# ---------------------------------------------------------------------------
# Passage / chapter analyze pipeline
# ---------------------------------------------------------------------------

def analyze_passage(
    passage: PassageData,
    llm: "LLMProvider",
) -> PassageAnalysisResult:
    """
    Passage or chapter analysis pipeline: passage → LLM → repair → verify.
    Returns a PassageAnalysisResult (summary + themes + citations, no questions).
    """
    if len(passage.text.strip()) < MIN_PASSAGE_CHARS:
        raise EmptyPassageError(
            f"Retrieved text for {passage.reference!r} is too short "
            f"({len(passage.text)} chars). Check the reference."
        )

    sys_prompt = build_passage_system_prompt()
    user_prompt = build_passage_user_prompt(passage)

    raw = llm.complete(system=sys_prompt, prompt=user_prompt, schema=PassageAnalysisResult)

    result = repair_and_validate(
        raw=raw,
        schema=PassageAnalysisResult,
        llm=llm,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
    )

    verify_citations(result, passage, mode=CitationMode.SINGLE_VERSE)

    return result


# ---------------------------------------------------------------------------
# Routed analyze() — dispatches by granularity
# ---------------------------------------------------------------------------

def analyze(
    reference: str,
    llm: "LLMProvider | None" = None,
) -> StudyGuideResult | PassageAnalysisResult | BookAnalysisResult:
    """
    Analyze a Bible reference, routing by detected granularity.

    - Passage (e.g. "John 3:16-21")   → PassageAnalysisResult
    - Chapter (e.g. "John 3")          → PassageAnalysisResult
    - Book    (e.g. "1 Corinthians")   → BookAnalysisResult

    Returns StudyGuideResult only when called from the legacy study-guide path
    (maintained for backward compatibility with Phase 1 tests and CLI).

    Raises:
        InvalidReferenceError:   unrecognised reference or book too large.
        EmptyPassageError:       retrieved text too short to analyze.
        CitationOutOfRangeError: LLM cited a verse outside the passage.
        AnalysisFailedError:     LLM output failed all repair/retry attempts.
    """
    if llm is None:
        from horeb.llm import ClaudeProvider
        llm = ClaudeProvider()

    ref, granularity = detect_granularity(reference)

    if granularity == Granularity.BOOK:
        return analyze_book(reference, llm)

    if granularity == Granularity.CHAPTER:
        passage = retrieve_chapter(ref.book, ref.start_chapter)
        return analyze_passage(passage, llm)

    # PASSAGE granularity — retrieve and run passage pipeline
    passage = retrieve_passage(reference)

    if len(passage.text.strip()) < MIN_PASSAGE_CHARS:
        raise EmptyPassageError(
            f"Retrieved text for {reference!r} is too short "
            f"({len(passage.text)} chars). Check the reference."
        )

    sys_prompt = build_passage_system_prompt()
    user_prompt_str = build_passage_user_prompt(passage)

    raw_response = llm.complete(system=sys_prompt, prompt=user_prompt_str, schema=PassageAnalysisResult)

    result = repair_and_validate(
        raw=raw_response,
        schema=PassageAnalysisResult,
        llm=llm,
        system_prompt=sys_prompt,
        user_prompt=user_prompt_str,
    )

    verify_citations(result, passage, mode=CitationMode.SINGLE_VERSE)

    return result


# ---------------------------------------------------------------------------
# Legacy study-guide pipeline (Phase 1 — kept for backward compat with tests)
# ---------------------------------------------------------------------------

def analyze_study_guide(reference: str, llm: "LLMProvider | None" = None) -> StudyGuideResult:
    """
    Phase 1 study guide pipeline: passage → LLM → repair → verify.
    Produces StudyGuideResult (summary + themes + entities + 5 questions).
    Called directly by Phase 1 tests via the engine.analyze alias below.
    """
    if llm is None:
        from horeb.llm import ClaudeProvider
        llm = ClaudeProvider()

    passage = retrieve_passage(reference)

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

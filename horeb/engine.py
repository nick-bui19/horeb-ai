from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import TYPE_CHECKING

import pythonbible as pb

from horeb.bible_text import (
    Granularity,
    MAX_BOOK_LLM_CALLS,
    Segment,
    _get_verse_text,
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
from horeb.parallels import score_similarity
from horeb.prompts import (
    SYSTEM_PROMPT,
    build_passage_system_prompt,
    build_passage_user_prompt,
    build_segment_system_prompt,
    build_segment_user_prompt,
    build_synthesis_system_prompt,
    build_synthesis_user_prompt,
    build_tag_system_prompt,
    build_tag_user_prompt,
    build_user_prompt,
)
from horeb.repair import repair_and_validate
from horeb.schemas import (
    AnalysisResult,
    BookAnalysisResult,
    PassageAnalysisResult,
    PassageData,
    SemanticTagResult,
    SegmentFailure,
    SegmentResult,
    SimilarityResult,
    SimilarOverlap,
    StudyGuideResult,
    TaggedCandidate,
)

if TYPE_CHECKING:
    from horeb.llm import LLMProvider

MIN_PASSAGE_CHARS: int = 20
SEGMENT_FAILURE_THRESHOLD: float = 0.3   # abort book if >30% of segments fail
_MAX_PARALLEL_WORKERS: int = 5            # concurrent segment LLM calls in book pipeline stage 1
_SYNTHESIS_MAX_TOKENS: int = 6144         # output budget for synthesis (supports 60-section outlines)
_MAX_TAG_CANDIDATES: int = 10             # hard cap on candidates sent to the 6A tagging LLM call


# ---------------------------------------------------------------------------
# Text normalisation for verbatim quote validation
# ---------------------------------------------------------------------------

_VERSE_LABEL_RE = re.compile(r"\[\d+:\d+\]\s*")


def _normalize_quote(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for substring comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_verse_labels(text: str) -> str:
    """Remove [chapter:verse] labels from labelled passage text before quote matching.

    Labels like [3:17] become '317' after _normalize_quote strips punctuation,
    which breaks multi-verse quote matching since the LLM's verbatim quotes don't
    include labels.
    """
    return _VERSE_LABEL_RE.sub("", text)


def _strip_candidate_label(candidate_text: str) -> str:
    """Strip the [ch:v] prefix from a single-verse candidate text."""
    return _VERSE_LABEL_RE.sub("", candidate_text).strip()


def _best_seed_verse(seed_text: str, overlap_terms: list[str]) -> str:
    """Split seed text on [ch:v] labels; return the verse with the most overlap_terms.

    Falls back to the full stripped text if no labeled verses are found.
    """
    verse_texts = re.split(r"\[\d+:\d+\]\s*", seed_text)
    verse_texts = [v.strip() for v in verse_texts if v.strip()]

    if not verse_texts:
        return seed_text.strip()

    if len(verse_texts) == 1 or not overlap_terms:
        return verse_texts[0]

    overlap_set = {t.lower() for t in overlap_terms}

    def count_overlap(text: str) -> int:
        words = set(re.sub(r"[^\w\s]", "", text.lower()).split())
        return len(words & overlap_set)

    return max(verse_texts, key=count_overlap)


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
    segments: list[Segment] | None = None,
) -> None:
    """
    Verify that every outline section in the synthesis output is grounded in
    at least one validated segment result.

    If segments is provided, also validates that each outline section's
    start_verse and end_verse anchors are in valid "chapter:verse" format and
    that their chapter falls within the chapter range of the cited source_segments.

    Raises:
        CitationOutOfRangeError: if any outline section has empty source_segments,
            references a non-existent segment index, has malformed verse anchors,
            or has verse anchors outside the chapter range of cited source_segments.
    """
    valid_indices = {s.segment_index for s in segment_results}

    # Build chapter range lookup if segments list is provided
    seg_chapter_range: dict[int, tuple[int, int]] = {}
    if segments:
        for s in segments:
            seg_chapter_range[s.segment_index] = (s.start_chapter, s.end_chapter)

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

        # Validate verse anchor format and chapter range against source segments
        if seg_chapter_range:
            valid_chapters: set[int] = set()
            for seg_idx in section.source_segments:
                if seg_idx in seg_chapter_range:
                    start_ch, end_ch = seg_chapter_range[seg_idx]
                    valid_chapters.update(range(start_ch, end_ch + 1))

            for anchor_field, anchor in [
                ("start_verse", section.start_verse),
                ("end_verse", section.end_verse),
            ]:
                if not anchor:
                    continue
                try:
                    parts = anchor.split(":")
                    if len(parts) != 2:
                        raise ValueError
                    chapter, verse = int(parts[0]), int(parts[1])
                    if chapter < 1 or verse < 1:
                        raise ValueError
                except ValueError:
                    raise CitationOutOfRangeError(
                        f"Outline section {i} ({section.title!r}) {anchor_field} "
                        f"{anchor!r} is not a valid chapter:verse coordinate."
                    )
                if valid_chapters and chapter not in valid_chapters:
                    raise CitationOutOfRangeError(
                        f"Outline section {i} ({section.title!r}) {anchor_field} "
                        f"{anchor!r} has chapter {chapter} outside the chapter range "
                        f"of its source segments (valid chapters: {sorted(valid_chapters)})."
                    )


# ---------------------------------------------------------------------------
# Book pipeline — segment worker (called concurrently from analyze_book stage 1)
# ---------------------------------------------------------------------------

def _run_segment(
    seg: "Segment",
    llm: "LLMProvider",
    seg_system_prompt: str,
) -> "tuple[SegmentResult | SegmentFailure, int]":
    """
    Analyze one book segment. Returns (result, llm_calls_made).

    Always returns a value — exceptions are caught and converted to SegmentFailure.
    Stateless and safe to call from multiple threads concurrently.
    """
    if len(seg.text.strip()) < MIN_PASSAGE_CHARS:
        return (
            SegmentFailure(
                segment_index=seg.segment_index,
                chapter_start=seg.start_chapter,
                chapter_end=seg.end_chapter,
                error="Segment text too short",
            ),
            0,
        )

    user_prompt = build_segment_user_prompt(seg.text, seg.reference, seg.segment_index)
    calls = 0
    try:
        raw = llm.complete(
            system=seg_system_prompt,
            prompt=user_prompt,
            schema=SegmentResult,
        )
        calls += 1

        seg_result, retry_calls = repair_and_validate(
            raw=raw,
            schema=SegmentResult,
            llm=llm,
            system_prompt=seg_system_prompt,
            user_prompt=user_prompt,
        )
        calls += retry_calls

        seg_result = seg_result.model_copy(update={"segment_index": seg.segment_index})

        seg_passage = PassageData(
            reference=seg.reference,
            book=seg.book.value,
            start_chapter=seg.start_chapter,
            start_verse=seg.start_verse,
            end_chapter=seg.end_chapter,
            end_verse=seg.end_verse,
            text=seg.text,
            context_before=None,
            context_after=None,
        )
        verify_citations(seg_result, seg_passage, mode=CitationMode.SINGLE_VERSE)

        return seg_result, calls

    except Exception as exc:
        return (
            SegmentFailure(
                segment_index=seg.segment_index,
                chapter_start=seg.start_chapter,
                chapter_end=seg.end_chapter,
                error=str(exc),
            ),
            calls,
        )


# ---------------------------------------------------------------------------
# Book pipeline — two-stage analyze_book()
# ---------------------------------------------------------------------------

def analyze_book(book_name: str, llm: "LLMProvider") -> BookAnalysisResult:
    """
    Two-stage book analysis pipeline.

    Stage 1: Deterministically segment the book. For each segment, retrieve
             text and call the LLM to produce a SegmentResult. Segment
             citations are post-validated against the segment's verse range.
             Collect SegmentResult | SegmentFailure outcomes.

    Stage 2: If segment failure rate is below SEGMENT_FAILURE_THRESHOLD,
             build a synthesis prompt from validated SegmentResults (with
             explicit gap markers for failures and verbatim cited verse texts
             as grounding anchors) and call the LLM to produce a
             BookAnalysisResult. Synthesis output is verified for source_segment
             grounding and verse anchor validity.

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

    # Pre-flight ceiling: reserve 2 calls for synthesis + 1 retry
    max_processable = (MAX_BOOK_LLM_CALLS - 2) // 2
    if len(segments) > max_processable:
        print(
            f"[WARN] {len(segments)} segments exceeds processable limit "
            f"({max_processable}) for MAX_BOOK_LLM_CALLS={MAX_BOOK_LLM_CALLS}. "
            f"Last {len(segments) - max_processable} segment(s) will be skipped.",
            file=sys.stderr,
        )
        for seg in segments[max_processable:]:
            segment_failures.append(SegmentFailure(
                segment_index=seg.segment_index,
                chapter_start=seg.start_chapter,
                chapter_end=seg.end_chapter,
                error="MAX_BOOK_LLM_CALLS ceiling reached",
            ))
    segs_to_process = segments[:max_processable]

    # Stage 1: per-segment analysis (parallelized — calls are I/O-bound and independent)
    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL_WORKERS) as executor:
        future_map = {
            executor.submit(_run_segment, seg, llm, seg_system_prompt): seg
            for seg in segs_to_process
        }
        for future in as_completed(future_map):
            result, calls = future.result()
            total_llm_calls += calls
            if isinstance(result, SegmentResult):
                segment_results.append(result)
            else:
                segment_failures.append(result)

    # Restore deterministic ordering (as_completed yields in completion order)
    segment_results.sort(key=lambda r: r.segment_index)

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
    # Build cited verse text lookup — gives synthesis model direct text anchors
    verse_texts: dict[int, list[tuple[str, str]]] = {}
    for seg_result in segment_results:
        seg_texts: list[tuple[str, str]] = []
        for citation in seg_result.citations:
            if citation.verse_reference and ":" in citation.verse_reference:
                try:
                    ch_str, v_str = citation.verse_reference.split(":", 1)
                    ch, v = int(ch_str), int(v_str)
                    orig_seg = next(
                        (s for s in segments if s.segment_index == seg_result.segment_index),
                        None,
                    )
                    if orig_seg is not None:
                        text = _get_verse_text(orig_seg.book.value, ch, v)
                        if text:
                            seg_texts.append((citation.verse_reference, text))
                except (ValueError, AttributeError):
                    pass
        if seg_texts:
            verse_texts[seg_result.segment_index] = seg_texts

    syn_system_prompt = build_synthesis_system_prompt()
    syn_user_prompt = build_synthesis_user_prompt(
        segment_results, segment_failures, verse_texts=verse_texts
    )

    syn_chars = len(syn_system_prompt) + len(syn_user_prompt)
    print(
        f"[INFO] Synthesis prompt: {syn_chars:,} chars (~{syn_chars // 4:,} tokens), "
        f"max_output={_SYNTHESIS_MAX_TOKENS} tokens.",
        file=sys.stderr,
    )

    raw_synthesis = llm.complete(
        system=syn_system_prompt,
        prompt=syn_user_prompt,
        schema=BookAnalysisResult,
        max_tokens=_SYNTHESIS_MAX_TOKENS,
    )
    total_llm_calls += 1

    book_result, retry_calls = repair_and_validate(
        raw=raw_synthesis,
        schema=BookAnalysisResult,
        llm=llm,
        system_prompt=syn_system_prompt,
        user_prompt=syn_user_prompt,
        max_tokens=_SYNTHESIS_MAX_TOKENS,
    )
    total_llm_calls += retry_calls

    # Stamp failed segment indices into result
    if segment_failures:
        book_result = book_result.model_copy(
            update={"failed_segments": [f.segment_index for f in segment_failures]}
        )

    # Verify synthesis grounding — source_segments + verse anchor range check
    verify_synthesis_grounding(book_result, segment_results, segments)

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

    result, _ = repair_and_validate(
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

    # PASSAGE granularity — delegate to analyze_passage
    passage = retrieve_passage(reference)
    return analyze_passage(passage, llm)


# ---------------------------------------------------------------------------
# find_similar — 6A evidence tagging helpers
# ---------------------------------------------------------------------------

def _validate_tag_result(
    tag_result: SemanticTagResult,
    tfidf_lookup: "dict[str, set[str]]",
) -> list[TaggedCandidate]:
    """
    Post-validate a SemanticTagResult against the TF-IDF candidate set.

    Rules (both must pass; failing entries are silently dropped):
    1. candidate_ref must exactly match a key in tfidf_lookup.
    2. every term in justification_terms must be in tfidf_lookup[candidate_ref].

    Args:
        tag_result:    Parsed and schema-validated SemanticTagResult from the LLM.
        tfidf_lookup:  Mapping of {candidate_ref: set(overlap_terms)} from the scorer.

    Returns:
        List of TaggedCandidate entries that passed both validation rules.
        May be empty if all entries fail.
    """
    valid: list[TaggedCandidate] = []
    for entry in tag_result.candidates:
        if entry.candidate_ref not in tfidf_lookup:
            continue
        allowed_terms = tfidf_lookup[entry.candidate_ref]
        if not set(entry.justification_terms).issubset(allowed_terms):
            continue
        valid.append(entry)
    return valid


def tag_candidates(
    seed_passage: PassageData,
    candidates: "list",
    llm: "LLMProvider",
) -> list[TaggedCandidate]:
    """
    Assign a 6A evidence tag to each TF-IDF candidate using one LLM call.

    At most _MAX_TAG_CANDIDATES candidates are sent to the LLM. If the
    candidate list exceeds this limit, an INFO message is printed to stderr
    and only the top-ranked candidates are tagged; the rest receive no tag.

    Args:
        seed_passage: The retrieved seed PassageData.
        candidates:   CandidateMatch list from score_similarity() (ordered by score desc).
        llm:          LLMProvider instance for the tagging call.

    Returns:
        List of validated TaggedCandidate entries. May be shorter than the
        input candidate list if some entries fail post-validation, or empty
        if the LLM call fails entirely.
    """
    if not candidates:
        return []

    to_tag = candidates[:_MAX_TAG_CANDIDATES]
    if len(candidates) > _MAX_TAG_CANDIDATES:
        print(
            f"[INFO] --tags: capping LLM input at {_MAX_TAG_CANDIDATES} of "
            f"{len(candidates)} candidates. Remaining candidates will have no tag.",
            file=sys.stderr,
        )

    # Build lookup for post-validation: {candidate_ref: set(overlap_terms)}
    tfidf_lookup: dict[str, set[str]] = {
        c.reference: set(c.overlap_terms) for c in to_tag
    }

    tag_sys = build_tag_system_prompt()
    tag_user = build_tag_user_prompt(seed_passage.text, seed_passage.reference, to_tag)

    try:
        raw = llm.complete(system=tag_sys, prompt=tag_user, schema=SemanticTagResult)
        tag_result, _ = repair_and_validate(
            raw=raw,
            schema=SemanticTagResult,
            llm=llm,
            system_prompt=tag_sys,
            user_prompt=tag_user,
        )
        return _validate_tag_result(tag_result, tfidf_lookup)
    except Exception as exc:
        print(
            f"[WARN] --tags: tagging call failed ({exc}). Returning untagged results.",
            file=sys.stderr,
        )
        return []


# ---------------------------------------------------------------------------
# find_similar pipeline
# ---------------------------------------------------------------------------

def find_similar(
    reference: str,
    scope_book: str | None = None,
    top_n: int = 10,
    tags: bool = False,
    llm: "LLMProvider | None" = None,
) -> SimilarityResult:
    """
    Find passages similar to the seed reference using TF-IDF scoring.

    By default fully deterministic — no LLM calls are made.
    When tags=True, a single LLM call assigns a 6A evidence tag to the top
    candidates (up to _MAX_TAG_CANDIDATES) after TF-IDF scoring completes.

    Stage 1: Retrieve seed passage.
    Stage 2: Resolve optional scope_book.
    Stage 3: TF-IDF scoring → scored_candidates.
    Stage 4: Build SimilarOverlap from scorer data directly.
    Stage 5: (tags=True only) tag_candidates() → stamp tag + justification_terms.

    Raises:
        InvalidReferenceError: unrecognised reference or scope_book.
        EmptyPassageError:     seed passage text too short.
    """
    # Retrieve seed passage
    seed_passage = retrieve_passage(reference)
    if len(seed_passage.text.strip()) < MIN_PASSAGE_CHARS:
        raise EmptyPassageError(
            f"Seed passage {reference!r} is too short to find similarities."
        )

    # Resolve optional scope book
    scope_pb_book: pb.Book | None = None
    if scope_book is not None:
        try:
            refs = pb.get_references(scope_book)
            if not refs:
                raise InvalidReferenceError(f"Could not find scope book: {scope_book!r}")
            scope_pb_book = refs[0].book
        except InvalidReferenceError:
            raise
        except Exception as exc:
            raise InvalidReferenceError(f"Could not parse scope book: {scope_book!r}") from exc

    # Deterministic TF-IDF scoring (no LLM)
    scored_candidates = score_similarity(seed_passage, scope_book=scope_pb_book, top_n=top_n)
    if not scored_candidates:
        return SimilarityResult(seed_ref=reference, candidates=[])

    candidates: list[SimilarOverlap] = []
    for c in scored_candidates:
        verbatim_candidate_quote = _strip_candidate_label(c.text)
        verbatim_seed_quote = _best_seed_verse(seed_passage.text, c.overlap_terms)
        candidates.append(SimilarOverlap(
            candidate_ref=c.reference,
            verbatim_seed_quote=verbatim_seed_quote,
            verbatim_candidate_quote=verbatim_candidate_quote,
            overlap_terms=c.overlap_terms,
            similarity_score=c.similarity_score,
        ))

    # Stage 5 (optional): 6A evidence tagging — stamps tag + justification_terms
    if tags:
        if llm is None:
            from horeb.llm import ClaudeProvider
            llm = ClaudeProvider()
        tagged = tag_candidates(seed_passage, scored_candidates, llm)
        # Build lookup: {candidate_ref: TaggedCandidate} for O(1) stamping
        tag_lookup: dict[str, TaggedCandidate] = {t.candidate_ref: t for t in tagged}
        candidates = [
            c.model_copy(update={
                "tag": tag_lookup[c.candidate_ref].tag,
                "justification_terms": tag_lookup[c.candidate_ref].justification_terms,
            }) if c.candidate_ref in tag_lookup else c
            for c in candidates
        ]

    return SimilarityResult(seed_ref=reference, candidates=candidates)


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

    result, _ = repair_and_validate(
        raw=raw_response,
        schema=StudyGuideResult,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    verify_citations(result, passage)

    return result
"""
Prompt builders for every command and stage in Horeb.

Structure:
  _GROUNDING_PREAMBLE      — shared grounding + citation rules (included in every system prompt)
  _REFUSAL_INSTRUCTION     — shared null / low_confidence_fields rules

  build_passage_system_prompt()       → str
  build_passage_user_prompt(passage)  → str   (passage / chapter analyze)

  build_segment_system_prompt()             → str
  build_segment_user_prompt(text, ref, idx) → str   (book pipeline stage 1)

  build_synthesis_system_prompt()                            → str
  build_synthesis_user_prompt(segments, failed_segments)    → str   (book pipeline stage 2)

  build_similarity_system_prompt()                           → str
  build_similarity_user_prompt(seed, candidates)            → str   (find_similar)

  # Phase 1 backward-compat aliases (used by engine.py until Task 8 routing lands)
  SYSTEM_PROMPT     = build_passage_system_prompt()  (module-level constant)
  build_user_prompt = build_passage_user_prompt      (function alias)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from horeb.schemas import PassageData, SegmentFailure, SegmentResult


# ---------------------------------------------------------------------------
# Shared preamble blocks — included in every system prompt
# ---------------------------------------------------------------------------

_GROUNDING_PREAMBLE: str = """\
GROUNDING RULES:
- Analyse ONLY the text provided in the marked sections below.
- Never introduce information from outside the provided text — no outside commentary,
  theological tradition, or cross-references not present in the given text.
- If you cannot determine a value from the provided text alone, return null for that
  field and add the field name to low_confidence_fields.\
"""

_CITATION_RULES: str = """\
CITATION RULES:
- Every verse_reference you provide must be a verse that appears in the PASSAGE section.
- Do not cite verses from CONTEXT sections or from segments not provided to you.
- Use the format "chapter:verse" (e.g., "3:16") matching the [chapter:verse] labels in the text.\
"""

_REFUSAL_INSTRUCTION: str = """\
REFUSAL RULES:
- If a field cannot be grounded in the provided text, return null for that field.
- Add any null field name to the low_confidence_fields list.
- Do not speculate, infer, or fill gaps with general theological knowledge.\
"""

_TOOL_INSTRUCTION: str = (
    "Use the provided tool to return your structured response. "
    "Do not return plain text."
)


# ---------------------------------------------------------------------------
# Passage / chapter analyze
# ---------------------------------------------------------------------------

def build_passage_system_prompt() -> str:
    """System prompt for passage-level and chapter-level analyze."""
    return "\n\n".join([
        "You are a Bible passage analysis engine with strict grounding requirements.",
        _GROUNDING_PREAMBLE,
        _CITATION_RULES,
        _REFUSAL_INSTRUCTION,
        "OUTPUT RULES:\n"
        "- Provide exactly 3 summary bullet points — no more, no fewer.\n"
        "- Each summary point must be grounded in a specific statement from the passage.\n"
        "- Provide up to 5 key themes drawn only from the passage text.\n"
        "- Provide verse-level citations only for verses that appear in the PASSAGE section.",
        _TOOL_INSTRUCTION,
    ])


def build_passage_user_prompt(passage: "PassageData") -> str:
    """
    Build the user-facing prompt from a retrieved passage.

    Context sections are labelled explicitly and excluded from analysis and
    citation to prevent the model from citing out-of-range verses.
    """
    parts: list[str] = []

    if passage.context_before is not None:
        parts.append("CONTEXT (preceding verses — do not analyse or cite these):")
        parts.append(passage.context_before)
        parts.append("")

    parts.append(f"PASSAGE ({passage.reference}):")
    parts.append(passage.text)

    if passage.context_after is not None:
        parts.append("")
        parts.append("CONTEXT (following verses — do not analyse or cite these):")
        parts.append(passage.context_after)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Book pipeline — stage 1: per-segment analysis
# ---------------------------------------------------------------------------

def build_segment_system_prompt() -> str:
    """System prompt for a single book segment (one chapter or verse window)."""
    return "\n\n".join([
        "You are a Bible book analysis engine processing one segment of a larger book.",
        _GROUNDING_PREAMBLE,
        _CITATION_RULES,
        _REFUSAL_INSTRUCTION,
        "OUTPUT RULES:\n"
        "- Provide exactly 3 summary bullet points for this segment only.\n"
        "- Provide an outline_label: a short title for this segment (≤8 words).\n"
        "- Provide up to 3 key themes from this segment's text only.\n"
        "- Provide up to 5 verse citations from this segment only.\n"
        "- Do not reference content from other chapters or the book as a whole.",
        _TOOL_INSTRUCTION,
    ])


def build_segment_user_prompt(text: str, reference: str, segment_index: int) -> str:
    """Build the user prompt for a single book segment."""
    return "\n".join([
        f"SEGMENT {segment_index} ({reference}):",
        text,
    ])


# ---------------------------------------------------------------------------
# Book pipeline — stage 2: synthesis
# ---------------------------------------------------------------------------

def build_synthesis_system_prompt() -> str:
    """
    System prompt for book-level synthesis.

    Strictly limits the model to reorganising and labelling validated segment
    outputs — it must not introduce claims beyond what the segments contain.
    """
    return "\n\n".join([
        "You are a Bible book synthesis engine. Your input is a set of validated "
        "segment analyses produced in an earlier stage.",
        "SYNTHESIS GROUNDING RULES:\n"
        "- You may ONLY reorganize, combine, and label information from the numbered "
        "segment summaries and themes below.\n"
        "- Do NOT add interpretations, connections, or conclusions not explicitly stated "
        "in the segment summaries or themes.\n"
        "- Do NOT speculate about segments marked as FAILED.\n"
        "- Every outline section must declare the source_segments it draws from "
        "(use the segment index numbers provided).\n"
        "- An outline section with no valid source_segments is not permitted.",
        _REFUSAL_INSTRUCTION,
        "OUTPUT RULES:\n"
        "- Produce a book outline: each section has a title, start_verse, end_verse, "
        "and source_segments list.\n"
        "- Produce exactly 3 book-level summary bullet points drawn only from segment summaries.\n"
        "- Produce up to 5 book-level themes drawn only from segment themes.\n"
        "- Use the verse anchor format \"chapter:verse\" (e.g. \"1:1\").",
        _TOOL_INSTRUCTION,
    ])


def build_synthesis_user_prompt(
    segments: "list[SegmentResult]",
    failed_segments: "list[SegmentFailure]",
) -> str:
    """
    Build synthesis input from validated segment outputs only.

    Failed segments are included as explicit gap markers so the synthesis
    model cannot infer or fill in their content.
    """
    # Build a lookup so we can interleave failures in index order
    failed_by_index: dict[int, "SegmentFailure"] = {f.segment_index: f for f in failed_segments}
    all_indices = sorted(
        {s.segment_index for s in segments} | set(failed_by_index.keys())
    )

    parts: list[str] = ["SEGMENT ANALYSES (use only this information for synthesis):"]

    for idx in all_indices:
        if idx in failed_by_index:
            f = failed_by_index[idx]
            parts.append(
                f"\n[Segment {idx}: Chapters {f.chapter_start}–{f.chapter_end} "
                f"— ANALYSIS FAILED, content unavailable. "
                f"Do not speculate about or fill in this segment.]"
            )
        else:
            seg = next(s for s in segments if s.segment_index == idx)
            themes_str = ", ".join(seg.key_themes) if seg.key_themes else "(none)"
            citations_str = (
                ", ".join(c.verse_reference for c in seg.citations)
                if seg.citations else "(none)"
            )
            summary_str = "\n  ".join(f"- {s}" for s in seg.summary)
            parts.append(
                f"\n[Segment {idx}: {seg.outline_label}]\n"
                f"Summary:\n  {summary_str}\n"
                f"Themes: {themes_str}\n"
                f"Citations: {citations_str}"
            )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# find_similar
# ---------------------------------------------------------------------------

def build_similarity_system_prompt() -> str:
    """System prompt for find_similar overlap explanation."""
    return "\n\n".join([
        "You are a Bible passage similarity engine. "
        "You have been given a seed passage and a list of candidate similar passages.",
        "SIMILARITY GROUNDING RULES:\n"
        "- For each candidate, you must quote the EXACT text from the seed passage "
        "that overlaps with the candidate (verbatim_seed_quote).\n"
        "- You must quote the EXACT text from the candidate passage that overlaps "
        "with the seed (verbatim_candidate_quote).\n"
        "- Both quotes must appear word-for-word in the texts provided to you.\n"
        "- Do NOT assert theological parallels, interpretive connections, or thematic "
        "similarities that are not evidenced by shared vocabulary in the provided texts.\n"
        "- Do NOT invent passages or references not in the candidate list.",
        _REFUSAL_INSTRUCTION,
        _TOOL_INSTRUCTION,
    ])


def build_similarity_user_prompt(
    seed_text: str,
    seed_ref: str,
    candidates: list[tuple[str, str, list[str]]],
) -> str:
    """
    Build the find_similar user prompt.

    Args:
        seed_text:  Full text of the seed passage with [chapter:verse] labels.
        seed_ref:   Reference string for the seed passage (e.g. "John 3:16-21").
        candidates: List of (reference, text, matched_terms) tuples from the scorer.
    """
    parts: list[str] = [
        f"SEED PASSAGE ({seed_ref}):",
        seed_text,
        "",
        "CANDIDATE PASSAGES (ranked by vocabulary overlap):",
    ]

    for i, (ref, text, terms) in enumerate(candidates, 1):
        terms_str = ", ".join(f'"{t}"' for t in terms[:10])
        parts.append(f"\n[Candidate {i}: {ref}]")
        parts.append(f"Matched terms: {terms_str}")
        parts.append(text)

    parts.append(
        "\nFor each candidate, extract the verbatim overlapping text from both "
        "the seed and the candidate. Return results using the provided tool."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Phase 1 backward-compat aliases
# Engine.py uses these until Task 8 routing lands.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = build_passage_system_prompt()
build_user_prompt = build_passage_user_prompt

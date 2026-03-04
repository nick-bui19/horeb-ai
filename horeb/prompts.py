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

  build_tag_system_prompt()                                 → str
  build_tag_user_prompt(seed_text, seed_ref, candidates)   → str   (find_similar --tags / 6A)

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
    verse_texts: "dict[int, list[tuple[str, str]]] | None" = None,
) -> str:
    """
    Build synthesis input from validated segment outputs only.

    Failed segments are included as explicit gap markers so the synthesis
    model cannot infer or fill in their content.

    Args:
        segments:        Validated SegmentResult objects from stage 1.
        failed_segments: SegmentFailure records for gap markers.
        verse_texts:     Optional mapping of segment_index → list of (verse_ref, text)
                         tuples. When provided, the verbatim text of each segment's cited
                         verses is appended to its block, giving the synthesis model a
                         direct text anchor rather than summaries alone.
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
            seg_block = (
                f"\n[Segment {idx}: {seg.outline_label}]\n"
                f"Summary:\n  {summary_str}\n"
                f"Themes: {themes_str}\n"
                f"Citations: {citations_str}"
            )
            if verse_texts and idx in verse_texts:
                verse_lines = "\n  ".join(
                    f"[{ref}] {text}" for ref, text in verse_texts[idx]
                )
                seg_block += f"\nCited verse texts:\n  {verse_lines}"
            parts.append(seg_block)

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
        "- Both quotes must be a SINGLE CONTIGUOUS span of text exactly as it appears "
        "in the provided passage — do NOT splice, join, or combine fragments from "
        "different parts of the text.\n"
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
        "\nFor each candidate, extract a single contiguous verbatim span from both "
        "the seed and the candidate that best demonstrates the shared vocabulary. "
        "Do NOT splice fragments. Return results using the provided tool."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# find_similar — 6A evidence tagging
# ---------------------------------------------------------------------------

_EVIDENCE_TAG_RULES: str = """\
EVIDENCE TAG RULES:
- You must classify each candidate using ONLY one of these five tags:
  * shared_phrase         — seed and candidate share a multi-word phrase verbatim
  * shared_rare_terms     — seed and candidate share uncommon/specific vocabulary
  * shared_imagery_terms  — seed and candidate share concrete image words (e.g. water, light, shepherd)
  * shared_speech_act_terms — seed and candidate share words denoting speech or command
  * weak_match            — some overlap exists but does not clearly fit the above
- justification_terms MUST be drawn ONLY from the overlap_terms list provided for each candidate.
- Do NOT introduce new terms not present in overlap_terms.
- Do NOT assert theological meaning, doctrinal equivalence, or interpretive claims.
- Do NOT invent candidate references not in the provided list.\
"""


def build_tag_system_prompt() -> str:
    """System prompt for the 6A evidence-tagging call."""
    return "\n\n".join([
        "You are a textual evidence classifier for Bible passage similarity results. "
        "Your only role is to classify vocabulary overlap between passages.",
        _EVIDENCE_TAG_RULES,
        _REFUSAL_INSTRUCTION,
        _TOOL_INSTRUCTION,
    ])


def build_tag_user_prompt(
    seed_text: str,
    seed_ref: str,
    candidates: "list",
) -> str:
    """
    Build the tagging user prompt.

    Args:
        seed_text:  Full text of the seed passage with [chapter:verse] labels.
        seed_ref:   Reference string for the seed (e.g. "John 3:16-21").
        candidates: List of CandidateMatch objects from the TF-IDF scorer.
    """
    parts: list[str] = [
        f"SEED PASSAGE ({seed_ref}):",
        seed_text,
        "",
        "CANDIDATES TO CLASSIFY (each includes its overlap_terms — use ONLY these for justification_terms):",
    ]

    for c in candidates:
        terms_str = ", ".join(f'"{t}"' for t in c.overlap_terms)
        parts.append(f"\n[{c.reference}]")
        parts.append(f"overlap_terms: {terms_str}")
        parts.append(c.text)

    parts.append(
        "\nFor each candidate, assign one tag and list justification_terms drawn "
        "strictly from its overlap_terms. Return results using the provided tool."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Phase 1 backward-compat aliases
# Engine.py uses these until Task 8 routing lands.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = build_passage_system_prompt()
build_user_prompt = build_passage_user_prompt

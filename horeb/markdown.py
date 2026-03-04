"""
Markdown rendering for all Horeb result types.

Public API:
    extract_sections(result, reference) -> ResultSections
        Shared extractor — both plain-text and markdown renderers consume this.
        Single place to update when schemas change.

    render_analysis_md(result, reference) -> str
        Produce CommonMark markdown for any analyze result.

    render_similar_md(result) -> str
        Produce CommonMark markdown for a SimilarityResult.

Design:
    extract_sections() is a pure function with no I/O. It normalises the
    union of all result types into a ResultSections dataclass so that
    the rendering functions are thin formatters with no branching on result type.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from horeb.schemas import (
        BookAnalysisResult,
        OutlineSection,
        PassageAnalysisResult,
        SimilarityResult,
        StudyGuideResult,
    )

_SUMMARY_WRAP: int = 88
_OUTLINE_SUMMARY_MAX: int = 60  # truncate long outline section summaries in the table


# ---------------------------------------------------------------------------
# Shared extracted structure
# ---------------------------------------------------------------------------

@dataclass
class ResultSections:
    """
    Normalised content extracted from any analyze result type.

    All fields are populated from the result — fields not applicable to a
    given result type are left as empty lists or None.
    """
    title: str
    summary: list[str]
    themes: list[str]
    # PassageAnalysisResult / SegmentResult citations: (verse_ref, quoted_text | None)
    citations: list[tuple[str, str | None]] = field(default_factory=list)
    # StudyGuideResult questions: (type_value, text, verse_ref | None)
    questions: list[tuple[str, str, str | None]] = field(default_factory=list)
    # BookAnalysisResult outline sections
    outline: list["OutlineSection"] = field(default_factory=list)
    failed_segments: list[int] = field(default_factory=list)
    low_confidence: list[str] = field(default_factory=list)


def extract_sections(
    result: "StudyGuideResult | PassageAnalysisResult | BookAnalysisResult",
    reference: str,
) -> ResultSections:
    """
    Extract all renderable content from any analyze result into ResultSections.

    This is the single source of truth for what gets rendered — both
    plain-text printing and markdown rendering consume this output.
    """
    from horeb.schemas import BookAnalysisResult, PassageAnalysisResult, StudyGuideResult

    sections = ResultSections(
        title=f"Analysis: {reference}",
        summary=list(result.summary),
        themes=list(result.key_themes) if result.key_themes else [],
        low_confidence=list(result.low_confidence_fields),
    )

    if isinstance(result, StudyGuideResult):
        if result.questions:
            sections.questions = [
                (q.type.value, q.text, q.verse_reference)
                for q in result.questions
            ]

    if isinstance(result, (PassageAnalysisResult, StudyGuideResult)):
        # StudyGuideResult has no citations field; PassageAnalysisResult does
        if hasattr(result, "citations") and result.citations:
            sections.citations = [
                (c.verse_reference, c.quoted_text)
                for c in result.citations
            ]

    if isinstance(result, BookAnalysisResult):
        sections.outline = list(result.outline)
        sections.failed_segments = list(result.failed_segments)

    return sections


# ---------------------------------------------------------------------------
# Markdown renderer — analyze results
# ---------------------------------------------------------------------------

def render_analysis_md(
    result: "StudyGuideResult | PassageAnalysisResult | BookAnalysisResult",
    reference: str,
) -> str:
    """
    Render any analyze result as CommonMark markdown.

    Args:
        result:    Any analyze result type.
        reference: The original reference string (used as the document title).

    Returns:
        A markdown string ready to write to a file.
    """
    sections = extract_sections(result, reference)
    parts: list[str] = []

    parts.append(f"# {sections.title}\n")

    # Summary
    parts.append("## Summary\n")
    for sentence in sections.summary:
        wrapped = textwrap.fill(sentence, width=_SUMMARY_WRAP, subsequent_indent="  ")
        parts.append(f"- {wrapped}")
    parts.append("")

    # Themes
    parts.append("## Key Themes\n")
    if sections.themes:
        for theme in sections.themes[:5]:
            parts.append(f"- {theme}")
    else:
        parts.append("_(not determined from passage text)_")
    parts.append("")

    # Study questions (StudyGuideResult)
    if sections.questions:
        parts.append("## Study Questions\n")
        for i, (qtype, text, verse_ref) in enumerate(sections.questions, 1):
            parts.append(f"{i}. **[{qtype}]** {text}")
            if verse_ref:
                parts.append(f"   _(cf. {verse_ref})_")
        parts.append("")

    # Citations (PassageAnalysisResult)
    if sections.citations:
        parts.append("## Citations\n")
        for verse_ref, quoted_text in sections.citations:
            snippet = f" — _{quoted_text}_" if quoted_text else ""
            parts.append(f"- `{verse_ref}`{snippet}")
        parts.append("")

    # Outline (BookAnalysisResult)
    if sections.outline:
        parts.append("## Outline\n")
        parts.append("| Section | Range | Summary |")
        parts.append("|---------|-------|---------|")
        for section in sections.outline:
            title = section.title
            range_str = f"{section.start_verse}–{section.end_verse}"
            summary = section.summary or ""
            if len(summary) > _OUTLINE_SUMMARY_MAX:
                summary = summary[:_OUTLINE_SUMMARY_MAX - 1] + "…"
            parts.append(f"| {title} | {range_str} | {summary} |")
        parts.append("")

    # Failed segments notice
    if sections.failed_segments:
        count = len(sections.failed_segments)
        parts.append(
            f"> **Note:** {count} segment(s) could not be analyzed: "
            f"{sections.failed_segments}\n"
        )

    # Low confidence notice
    if sections.low_confidence:
        fields = ", ".join(sections.low_confidence)
        parts.append(f"> **Low confidence fields:** {fields}\n")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Markdown renderer — similarity results
# ---------------------------------------------------------------------------

def render_similar_md(result: "SimilarityResult") -> str:
    """
    Render a SimilarityResult as CommonMark markdown.

    Args:
        result: The SimilarityResult from find_similar().

    Returns:
        A markdown string ready to write to a file.
    """
    parts: list[str] = []
    parts.append(f"# Similar Passages: {result.seed_ref}\n")

    if not result.candidates:
        parts.append("_No similar passages found._\n")
        return "\n".join(parts)

    for i, c in enumerate(result.candidates, 1):
        parts.append(f"## {i}. {c.candidate_ref}\n")
        parts.append(f"**Score:** {c.similarity_score:.4f}\n")

        if c.tag is not None:
            if c.justification_terms:
                terms_str = ", ".join(f"`{t}`" for t in c.justification_terms)
                parts.append(f"**Tag:** {c.tag}  [{terms_str}]\n")
            else:
                parts.append(f"**Tag:** {c.tag}\n")

        if c.overlap_terms:
            terms_str = ", ".join(c.overlap_terms)
            parts.append(f"**Overlap:** {terms_str}\n")

        if c.verbatim_seed_quote:
            parts.append(f"**Seed:** \"{c.verbatim_seed_quote}\"\n")

        if c.verbatim_candidate_quote:
            parts.append(f"**Candidate:** \"{c.verbatim_candidate_quote}\"\n")

    return "\n".join(parts)

"""
Pure HTML rendering functions for the Horeb desktop app.
No Qt imports — these are plain string functions, fully testable without a display.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from horeb.markdown import extract_sections

if TYPE_CHECKING:
    from horeb.schemas import SimilarityResult

# Reuse the same verse-label regex from engine.py
_VERSE_LABEL_RE = re.compile(r"\[(\d+):(\d+)\]\s*")

# ---------------------------------------------------------------------------
# Chapter text → HTML
# ---------------------------------------------------------------------------

_CHAPTER_STYLE = (
    "font-family: Georgia, serif; font-size: 18px; line-height: 1.8; "
    "color: #1c1c1e; max-width: 680px; margin: 0 auto; padding: 20px 0;"
)


def chapter_text_to_html(text: str, chapter: int) -> str:
    """
    Convert raw chapter text (lines of "[ch:v] verse text") to HTML.

    Verse numbers become inline <sup> superscripts. The chapter number is
    rendered as a bold <h2> at the top.
    """
    lines = text.strip().splitlines()
    verse_spans: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = _VERSE_LABEL_RE.match(line)
        if m:
            verse_num = m.group(2)
            verse_text = line[m.end():]
            verse_text_escaped = _escape_html(verse_text)
            verse_spans.append(
                f'<sup style="font-size:11px; color:#8e8e93; '
                f'vertical-align:super; margin-right:3px;">{verse_num}</sup>'
                f'{verse_text_escaped} '
            )
        else:
            verse_spans.append(_escape_html(line) + " ")

    body = "".join(verse_spans)

    return (
        f'<div style="{_CHAPTER_STYLE}">'
        f'<h2 style="font-size:20px; font-weight:bold; margin-bottom:16px; '
        f'color:#1c1c1e;">Chapter {chapter}</h2>'
        f'<p style="margin:0; text-align:justify;">{body}</p>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Analysis result → HTML
# ---------------------------------------------------------------------------

_RESULT_STYLE = (
    "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
    "font-size: 14px; color: #1c1c1e; padding: 8px 0;"
)


def result_to_html(result: object, reference: str) -> str:
    """
    Render any analyze result (PassageAnalysisResult / BookAnalysisResult /
    StudyGuideResult) as HTML, reusing extract_sections() from horeb/markdown.py.
    """
    sections = extract_sections(result, reference)  # type: ignore[arg-type]
    parts: list[str] = []

    parts.append(f'<div style="{_RESULT_STYLE}">')

    # Title
    parts.append(
        f'<h3 style="margin:0 0 12px 0; font-size:15px; color:#1c1c1e;">'
        f'{_escape_html(sections.title)}</h3>'
    )

    # Summary
    parts.append('<p style="margin:0 0 4px 0; font-weight:bold; color:#636366; font-size:12px; text-transform:uppercase; letter-spacing:0.5px;">Summary</p>')
    parts.append('<ul style="margin:0 0 12px 0; padding-left:20px;">')
    for sentence in sections.summary:
        parts.append(f'<li style="margin-bottom:4px;">{_escape_html(sentence)}</li>')
    parts.append("</ul>")

    # Themes
    if sections.themes:
        parts.append('<p style="margin:0 0 4px 0; font-weight:bold; color:#636366; font-size:12px; text-transform:uppercase; letter-spacing:0.5px;">Key Themes</p>')
        parts.append('<ul style="margin:0 0 12px 0; padding-left:20px;">')
        for theme in sections.themes[:5]:
            parts.append(f'<li style="margin-bottom:2px;">{_escape_html(theme)}</li>')
        parts.append("</ul>")

    # Study questions (StudyGuideResult)
    if sections.questions:
        parts.append('<p style="margin:0 0 4px 0; font-weight:bold; color:#636366; font-size:12px; text-transform:uppercase; letter-spacing:0.5px;">Study Questions</p>')
        parts.append('<ol style="margin:0 0 12px 0; padding-left:20px;">')
        for qtype, text, verse_ref in sections.questions:
            ref_str = f' <span style="color:#8e8e93; font-size:12px;">({_escape_html(verse_ref)})</span>' if verse_ref else ""
            parts.append(
                f'<li style="margin-bottom:6px;">'
                f'<span style="background:#e8f5e9; color:#2e7d32; border-radius:3px; '
                f'padding:1px 5px; font-size:11px; margin-right:6px;">{_escape_html(qtype)}</span>'
                f'{_escape_html(text)}{ref_str}</li>'
            )
        parts.append("</ol>")

    # Citations
    if sections.citations:
        parts.append('<p style="margin:0 0 4px 0; font-weight:bold; color:#636366; font-size:12px; text-transform:uppercase; letter-spacing:0.5px;">Citations</p>')
        parts.append('<ul style="margin:0 0 12px 0; padding-left:20px;">')
        for verse_ref, quoted_text in sections.citations:
            quote = f' — <em>{_escape_html(quoted_text)}</em>' if quoted_text else ""
            parts.append(
                f'<li style="margin-bottom:4px;">'
                f'<code style="background:#f0f0f0; padding:1px 4px; border-radius:3px;">'
                f'{_escape_html(verse_ref)}</code>{quote}</li>'
            )
        parts.append("</ul>")

    # Outline table (BookAnalysisResult)
    if sections.outline:
        parts.append('<p style="margin:0 0 4px 0; font-weight:bold; color:#636366; font-size:12px; text-transform:uppercase; letter-spacing:0.5px;">Outline</p>')
        parts.append(
            '<table style="width:100%; border-collapse:collapse; margin-bottom:12px; font-size:13px;">'
            '<tr style="background:#f0f0f0;">'
            '<th style="text-align:left; padding:4px 8px; border-bottom:1px solid #d1d1d6;">Section</th>'
            '<th style="text-align:left; padding:4px 8px; border-bottom:1px solid #d1d1d6;">Range</th>'
            '<th style="text-align:left; padding:4px 8px; border-bottom:1px solid #d1d1d6;">Summary</th>'
            '</tr>'
        )
        for section in sections.outline:
            title = _escape_html(section.title)
            range_str = _escape_html(f"{section.start_verse}–{section.end_verse}")
            summary = section.summary or ""
            if len(summary) > 80:
                summary = summary[:79] + "…"
            summary_escaped = _escape_html(summary)
            parts.append(
                f'<tr>'
                f'<td style="padding:4px 8px; border-bottom:1px solid #f0f0f0;">{title}</td>'
                f'<td style="padding:4px 8px; border-bottom:1px solid #f0f0f0; white-space:nowrap;">{range_str}</td>'
                f'<td style="padding:4px 8px; border-bottom:1px solid #f0f0f0;">{summary_escaped}</td>'
                f'</tr>'
            )
        parts.append("</table>")

    # Failed segments notice
    if sections.failed_segments:
        count = len(sections.failed_segments)
        parts.append(
            f'<p style="background:#fff3e0; border-left:3px solid #ff9800; '
            f'padding:8px 12px; margin:0 0 8px 0; border-radius:0 4px 4px 0; font-size:13px;">'
            f'<strong>Note:</strong> {count} segment(s) could not be analyzed: '
            f'{_escape_html(str(sections.failed_segments))}</p>'
        )

    # Low confidence notice
    if sections.low_confidence:
        fields = ", ".join(sections.low_confidence)
        parts.append(
            f'<p style="background:#fce4ec; border-left:3px solid #e91e63; '
            f'padding:8px 12px; margin:0 0 8px 0; border-radius:0 4px 4px 0; font-size:13px;">'
            f'<strong>Low confidence:</strong> {_escape_html(fields)}</p>'
        )

    parts.append("</div>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Similarity result → HTML
# ---------------------------------------------------------------------------

def similar_to_html(result: "SimilarityResult") -> str:
    """
    Render a SimilarityResult as HTML — ranked list with scores, tags, quotes.
    """
    parts: list[str] = []
    parts.append(f'<div style="{_RESULT_STYLE}">')
    parts.append(
        f'<h3 style="margin:0 0 12px 0; font-size:15px; color:#1c1c1e;">'
        f'Similar Passages: {_escape_html(result.seed_ref)}</h3>'
    )

    if not result.candidates:
        parts.append('<p style="color:#8e8e93; font-style:italic;">No similar passages found.</p>')
        parts.append("</div>")
        return "\n".join(parts)

    for i, c in enumerate(result.candidates, 1):
        parts.append(
            f'<div style="margin-bottom:16px; padding:12px; background:#ffffff; '
            f'border-radius:8px; border:1px solid #e0e0e0;">'
        )
        # Header: rank + reference + score
        parts.append(
            f'<div style="display:flex; justify-content:space-between; margin-bottom:6px;">'
            f'<span style="font-weight:bold; font-size:14px;">'
            f'{i}. {_escape_html(c.candidate_ref)}</span>'
            f'<span style="color:#8e8e93; font-size:12px; font-family:monospace;">'
            f'score: {c.similarity_score:.4f}</span>'
            f'</div>'
        )

        # Tag
        if c.tag is not None:
            tag_str = c.tag.value if hasattr(c.tag, "value") else str(c.tag)
            parts.append(
                f'<span style="background:#e8f5e9; color:#2e7d32; border-radius:3px; '
                f'padding:2px 6px; font-size:12px; margin-right:6px;">{_escape_html(tag_str)}</span>'
            )
            if c.justification_terms:
                terms = ", ".join(f'<code style="font-size:11px;">{_escape_html(t)}</code>' for t in c.justification_terms)
                parts.append(f'<span style="color:#636366; font-size:12px;">[{terms}]</span>')
            parts.append("<br>")

        # Overlap terms
        if c.overlap_terms:
            terms = ", ".join(f'<code style="background:#f0f0f0; padding:1px 3px; border-radius:2px; font-size:11px;">{_escape_html(t)}</code>' for t in c.overlap_terms)
            parts.append(f'<p style="margin:6px 0 4px 0; font-size:12px; color:#636366;">Overlap: {terms}</p>')

        # Seed quote
        if c.verbatim_seed_quote:
            parts.append(
                f'<p style="margin:4px 0; font-size:13px; color:#636366;">'
                f'<strong>Seed:</strong> <em>"{_escape_html(c.verbatim_seed_quote)}"</em></p>'
            )

        # Candidate quote
        if c.verbatim_candidate_quote:
            parts.append(
                f'<p style="margin:4px 0; font-size:13px; color:#636366;">'
                f'<strong>Match:</strong> <em>"{_escape_html(c.verbatim_candidate_quote)}"</em></p>'
            )

        parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _escape_html(text: str) -> str:
    """Minimal HTML escaping for plain text insertion."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

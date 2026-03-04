import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()  # loads .env from the current working directory if present

from horeb.engine import analyze, find_similar as _find_similar
from horeb.errors import (
    AnalysisFailedError,
    CitationOutOfRangeError,
    EmptyPassageError,
    HorebError,
    InvalidReferenceError,
)
from horeb.markdown import extract_sections, render_analysis_md, render_similar_md
from horeb.schemas import (
    BookAnalysisResult,
    PassageAnalysisResult,
    SimilarityResult,
    StudyGuideResult,
)

# Exit codes — each HorebError subtype maps to a distinct code
# so callers (scripts, CI) can distinguish failure modes.
EXIT_INVALID_REFERENCE: int = 2
EXIT_EMPTY_PASSAGE: int = 3
EXIT_CITATION_OUT_OF_RANGE: int = 4
EXIT_ANALYSIS_FAILED: int = 5

app = typer.Typer(
    name="horeb",
    help="CLI-first AI engine for grounded Bible passage analysis.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command(name="analyze")
def analyze_cmd(
    reference: str = typer.Argument(
        ...,
        help="Bible reference: passage ('John 3:16-21'), chapter ('John 3'), or book ('Ruth')",
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Write result as Markdown to this file path (e.g. notes.md)",
        writable=True, resolve_path=True,
    ),
) -> None:
    """Analyse a Bible reference (passage, chapter, or whole book)."""
    try:
        result = analyze(reference)
        if output is not None:
            _write_markdown(render_analysis_md(result, reference), output)
        else:
            _print_result(result, reference)
    except InvalidReferenceError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=EXIT_INVALID_REFERENCE)
    except EmptyPassageError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=EXIT_EMPTY_PASSAGE)
    except CitationOutOfRangeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=EXIT_CITATION_OUT_OF_RANGE)
    except AnalysisFailedError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=EXIT_ANALYSIS_FAILED)
    # Unhandled HorebError subtypes (future additions) → exit 1
    except HorebError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)


@app.command(name="find-similar")
def find_similar_cmd(
    reference: str = typer.Argument(
        ..., help="Seed passage reference, e.g. 'John 3:16-21'"
    ),
    book: str | None = typer.Option(
        None, "--book", help="Limit search scope to a specific book (e.g. 'John')"
    ),
    top_n: int = typer.Option(
        10, "--top-n", help="Number of candidate passages to return"
    ),
    tags: bool = typer.Option(
        False, "--tags", help="Assign evidence tags to candidates via one LLM call (6A safe mode)"
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Write result as Markdown to this file path (e.g. similar.md)",
        writable=True, resolve_path=True,
    ),
) -> None:
    """Find passages similar to the seed reference using TF-IDF scoring."""
    try:
        result = _find_similar(reference, scope_book=book, top_n=top_n, tags=tags)
        if output is not None:
            _write_markdown(render_similar_md(result), output)
        else:
            _print_similar_result(result)
    except InvalidReferenceError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=EXIT_INVALID_REFERENCE)
    except EmptyPassageError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=EXIT_EMPTY_PASSAGE)
    except CitationOutOfRangeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=EXIT_CITATION_OUT_OF_RANGE)
    except AnalysisFailedError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=EXIT_ANALYSIS_FAILED)
    except HorebError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)


def _write_markdown(content: str, path: "Path") -> None:
    """Write markdown content to path, exiting cleanly on file errors."""
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        typer.echo(f"Error: could not write to {path}: {exc}", err=True)
        raise typer.Exit(code=1)


def _print_result(
    result: "StudyGuideResult | PassageAnalysisResult | BookAnalysisResult",
    reference: str = "",
) -> None:
    """Format and print any analyze result to stdout using extract_sections()."""
    sections = extract_sections(result, reference)

    print("=== SUMMARY ===")
    for sentence in sections.summary:
        print(f"  - {sentence}")

    print("\n=== KEY THEMES ===")
    if sections.themes:
        for theme in sections.themes[:5]:
            print(f"  - {theme}")
    else:
        print("  (not determined from passage text)")

    if sections.questions:
        print("\n=== STUDY QUESTIONS ===")
        for i, (qtype, text, verse_ref) in enumerate(sections.questions, 1):
            print(f"  {i}. [{qtype}] {text}")
            if verse_ref:
                print(f"     (cf. {verse_ref})")

    if sections.citations:
        print("\n=== CITATIONS ===")
        for verse_ref, quoted_text in sections.citations:
            snippet = f" — {quoted_text}" if quoted_text else ""
            print(f"  [{verse_ref}]{snippet}")

    if sections.outline:
        print("\n=== OUTLINE ===")
        for section in sections.outline:
            print(f"  {section.title} ({section.start_verse}–{section.end_verse})")
            if section.summary:
                print(f"    {section.summary}")

    if sections.failed_segments:
        count = len(sections.failed_segments)
        print(f"\n[NOTE] {count} segment(s) could not be analyzed: {sections.failed_segments}")

    if sections.low_confidence:
        fields = ", ".join(sections.low_confidence)
        print(f"\n[NOTE] Low confidence fields: {fields}")


def _print_similar_result(result: SimilarityResult) -> None:
    """Format and print a SimilarityResult to stdout."""
    if not result.candidates:
        print("No similar passages found.")
        return

    print(f"=== SIMILAR PASSAGES (seed: {result.seed_ref}) ===")
    for i, c in enumerate(result.candidates, 1):
        print(f"\n  {i}. {c.candidate_ref}  (score: {c.similarity_score:.4f})")
        if c.tag is not None:
            terms_str = (
                f"  [{', '.join(c.justification_terms)}]" if c.justification_terms else ""
            )
            print(f"     Tag:       {c.tag}{terms_str}")
        if c.overlap_terms:
            print(f"     Overlap: {', '.join(c.overlap_terms)}")
        if c.verbatim_seed_quote:
            print(f"     Seed:      \"{c.verbatim_seed_quote}\"")
        if c.verbatim_candidate_quote:
            print(f"     Candidate: \"{c.verbatim_candidate_quote}\"")

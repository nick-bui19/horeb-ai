import sys

import typer
from dotenv import load_dotenv

load_dotenv()  # loads .env from the current working directory if present

from horeb.engine import analyze
from horeb.errors import (
    AnalysisFailedError,
    CitationOutOfRangeError,
    EmptyPassageError,
    HorebError,
    InvalidReferenceError,
)
from horeb.schemas import AnalysisResult

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
)


@app.command()
def main(
    reference: str = typer.Argument(
        ..., help="Bible passage reference, e.g. 'John 3:16-21'"
    ),
) -> None:
    """Analyse a Bible passage and print a structured study guide."""
    try:
        result = analyze(reference)
        _print_result(result)
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


def _print_result(result: AnalysisResult) -> None:
    """Format and print an AnalysisResult to stdout."""
    print("=== SUMMARY ===")
    for sentence in result.summary:
        print(f"  - {sentence}")

    print("\n=== KEY THEMES ===")
    if result.key_themes:
        for theme in result.key_themes[:3]:
            print(f"  - {theme}")
    else:
        print("  (not determined from passage text)")

    if result.low_confidence_fields:
        fields = ", ".join(result.low_confidence_fields)
        print(f"\n[NOTE] Low confidence fields: {fields}")

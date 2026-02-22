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
from horeb.schemas import AnalysisResult, PassageData

if TYPE_CHECKING:
    from horeb.llm import LLMProvider

MIN_PASSAGE_CHARS: int = 20


def analyze(reference: str, llm: "LLMProvider | None" = None) -> AnalysisResult:
    """
    Full analysis pipeline: retrieve → validate → prompt → LLM → repair → verify.

    Args:
        reference: Bible passage reference string (e.g. "John 3:16-21").
        llm:       LLMProvider to use. Defaults to ClaudeProvider() if None.
                   Inject a FixtureLLMProvider in tests to avoid live API calls.

    Returns:
        Validated AnalysisResult with citations verified against the passage.

    Raises:
        InvalidReferenceError:    malformed reference or passage exceeds MAX_PASSAGE_VERSES.
        EmptyPassageError:        retrieved passage text is too short to analyse.
        CitationOutOfRangeError:  LLM cited a verse outside the retrieved passage.
        AnalysisFailedError:      LLM output failed all repair/retry attempts.
    """
    if llm is None:
        # Import here (not at module level) so that importing engine in tests
        # never touches ClaudeProvider and never reads the API key.
        from horeb.llm import ClaudeProvider
        llm = ClaudeProvider()

    passage = retrieve_passage(reference)  # raises InvalidReferenceError

    if len(passage.text.strip()) < MIN_PASSAGE_CHARS:
        raise EmptyPassageError(
            f"Retrieved text for {reference!r} is too short "
            f"({len(passage.text)} chars). Check the reference."
        )

    user_prompt = build_user_prompt(passage)
    raw_response = llm.complete(system=SYSTEM_PROMPT, prompt=user_prompt)

    result = repair_and_validate(raw=raw_response, llm=llm, prompt=user_prompt)

    verify_citations(result, passage)

    return result


def extract_verse_refs(result: AnalysisResult) -> list[str]:
    """Extract all verse_reference strings from the analysis result."""
    refs: list[str] = []
    for q in result.questions:
        if q.verse_reference is not None:
            refs.append(q.verse_reference)
    if result.named_entities is not None:
        for entity in result.named_entities:
            if entity.verse_reference is not None:
                refs.append(entity.verse_reference)
    return refs


def verify_citations(result: AnalysisResult, passage: PassageData) -> None:
    """
    Verify every cited verse reference falls within the retrieved passage range.

    Citation format from the LLM is "chapter:verse" (e.g. "3:16") matching
    the labels in the passage text. We also accept full references like "John 3:16".

    Raises:
        CitationOutOfRangeError: if any citation is outside the passage range
            or cannot be parsed.
    """
    cited_refs = extract_verse_refs(result)
    passage_start = (passage.start_chapter, passage.start_verse)
    passage_end = (passage.end_chapter, passage.end_verse)

    for ref_str in cited_refs:
        _check_citation(ref_str, passage, passage_start, passage_end)


def _check_citation(
    ref_str: str,
    passage: PassageData,
    passage_start: tuple[int, int],
    passage_end: tuple[int, int],
) -> None:
    """
    Parse a single citation string and verify it is within the passage range.

    Raises CitationOutOfRangeError if out of range or unparseable.
    """
    # Normalise short "chapter:verse" format to a full reference the model
    # can parse. The LLM is instructed to use "chapter:verse" format matching
    # the labels in the passage. We reconstruct the book name for pythonbible.
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
    except (ValueError, Exception) as exc:
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

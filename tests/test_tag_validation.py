"""
Pure unit tests for _validate_tag_result().

These tests exercise every post-validation rule for 6A evidence tagging:
1. candidate_ref must exactly match a key in the tfidf_lookup.
2. justification_terms must be a strict subset of that candidate's overlap_terms.

No LLM, no fixtures, no I/O. Constructs SemanticTagResult objects directly.
Follows the same pattern as test_engine.py's verify_citations tests.
"""
import pytest

from horeb.engine import _validate_tag_result
from horeb.schemas import EvidenceTag, SemanticTagResult, TaggedCandidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tag_result(candidates: list[dict]) -> SemanticTagResult:
    """Build a SemanticTagResult from a list of dicts for concise test construction."""
    return SemanticTagResult(
        candidates=[TaggedCandidate(**c) for c in candidates]
    )


def _make_lookup(entries: dict[str, list[str]]) -> dict[str, set[str]]:
    """Build a tfidf_lookup from {ref: [terms]} for concise test construction."""
    return {ref: set(terms) for ref, terms in entries.items()}


# ---------------------------------------------------------------------------
# Valid entries pass through unchanged
# ---------------------------------------------------------------------------

def test_valid_entry_passes_through() -> None:
    tag_result = _make_tag_result([
        {"candidate_ref": "John 3:17", "tag": "shared_phrase", "justification_terms": ["world", "save"]},
    ])
    lookup = _make_lookup({"John 3:17": ["world", "save", "love"]})

    valid = _validate_tag_result(tag_result, lookup)

    assert len(valid) == 1
    assert valid[0].candidate_ref == "John 3:17"
    assert valid[0].tag == "shared_phrase"
    assert valid[0].justification_terms == ["world", "save"]


def test_multiple_valid_entries_all_pass() -> None:
    tag_result = _make_tag_result([
        {"candidate_ref": "John 3:17", "tag": "shared_phrase", "justification_terms": ["world"]},
        {"candidate_ref": "John 3:18", "tag": "shared_rare_terms", "justification_terms": ["condemned"]},
        {"candidate_ref": "John 3:19", "tag": "shared_imagery_terms", "justification_terms": ["light"]},
    ])
    lookup = _make_lookup({
        "John 3:17": ["world", "save"],
        "John 3:18": ["condemned", "believe"],
        "John 3:19": ["light", "darkness"],
    })

    valid = _validate_tag_result(tag_result, lookup)

    assert len(valid) == 3


def test_empty_justification_terms_is_valid_subset() -> None:
    """Empty list is always a subset of any set — should pass."""
    tag_result = _make_tag_result([
        {"candidate_ref": "John 3:17", "tag": "weak_match", "justification_terms": []},
    ])
    lookup = _make_lookup({"John 3:17": ["world", "save"]})

    valid = _validate_tag_result(tag_result, lookup)

    assert len(valid) == 1


# ---------------------------------------------------------------------------
# Hallucinated candidate_ref — entry dropped
# ---------------------------------------------------------------------------

def test_hallucinated_ref_is_dropped() -> None:
    tag_result = _make_tag_result([
        {"candidate_ref": "John 99:99", "tag": "shared_phrase", "justification_terms": ["world"]},
    ])
    lookup = _make_lookup({"John 3:17": ["world", "save"]})

    valid = _validate_tag_result(tag_result, lookup)

    assert valid == []


def test_partial_hallucinated_ref_only_valid_survives() -> None:
    tag_result = _make_tag_result([
        {"candidate_ref": "John 3:17", "tag": "shared_phrase", "justification_terms": ["world"]},
        {"candidate_ref": "Revelation 1:1", "tag": "weak_match", "justification_terms": []},
    ])
    lookup = _make_lookup({"John 3:17": ["world", "save"]})

    valid = _validate_tag_result(tag_result, lookup)

    assert len(valid) == 1
    assert valid[0].candidate_ref == "John 3:17"


def test_all_hallucinated_refs_returns_empty() -> None:
    tag_result = _make_tag_result([
        {"candidate_ref": "Genesis 999:1", "tag": "weak_match", "justification_terms": []},
        {"candidate_ref": "Revelation 99:99", "tag": "weak_match", "justification_terms": []},
    ])
    lookup = _make_lookup({"John 3:17": ["world"]})

    valid = _validate_tag_result(tag_result, lookup)

    assert valid == []


# ---------------------------------------------------------------------------
# justification_terms not a subset of overlap_terms — entry dropped
# ---------------------------------------------------------------------------

def test_invented_justification_term_is_dropped() -> None:
    tag_result = _make_tag_result([
        {
            "candidate_ref": "John 3:17",
            "tag": "shared_rare_terms",
            "justification_terms": ["atonement", "covenant"],  # not in overlap_terms
        },
    ])
    lookup = _make_lookup({"John 3:17": ["world", "save"]})

    valid = _validate_tag_result(tag_result, lookup)

    assert valid == []


def test_partial_invented_terms_drops_entry() -> None:
    """Even one non-subset term drops the entire entry."""
    tag_result = _make_tag_result([
        {
            "candidate_ref": "John 3:17",
            "tag": "shared_rare_terms",
            "justification_terms": ["world", "grace"],  # "world" ok, "grace" not in overlap
        },
    ])
    lookup = _make_lookup({"John 3:17": ["world", "save"]})

    valid = _validate_tag_result(tag_result, lookup)

    assert valid == []


def test_mixed_valid_invalid_terms_only_valid_survives() -> None:
    tag_result = _make_tag_result([
        {
            "candidate_ref": "John 3:17",
            "tag": "shared_phrase",
            "justification_terms": ["world"],  # valid
        },
        {
            "candidate_ref": "John 3:18",
            "tag": "shared_rare_terms",
            "justification_terms": ["grace"],  # not in overlap_terms for 3:18
        },
    ])
    lookup = _make_lookup({
        "John 3:17": ["world", "save"],
        "John 3:18": ["condemned", "believe"],
    })

    valid = _validate_tag_result(tag_result, lookup)

    assert len(valid) == 1
    assert valid[0].candidate_ref == "John 3:17"


# ---------------------------------------------------------------------------
# Empty SemanticTagResult
# ---------------------------------------------------------------------------

def test_empty_candidates_returns_empty() -> None:
    tag_result = SemanticTagResult(candidates=[])
    lookup = _make_lookup({"John 3:17": ["world"]})

    valid = _validate_tag_result(tag_result, lookup)

    assert valid == []


def test_empty_lookup_drops_all_entries() -> None:
    tag_result = _make_tag_result([
        {"candidate_ref": "John 3:17", "tag": "weak_match", "justification_terms": []},
    ])
    lookup: dict[str, set[str]] = {}

    valid = _validate_tag_result(tag_result, lookup)

    assert valid == []


# ---------------------------------------------------------------------------
# All five valid tags are accepted by the schema
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag", [
    "shared_phrase",
    "shared_rare_terms",
    "shared_imagery_terms",
    "shared_speech_act_terms",
    "weak_match",
])
def test_all_valid_tags_accepted(tag: str) -> None:
    tag_result = _make_tag_result([
        {"candidate_ref": "John 3:17", "tag": tag, "justification_terms": ["world"]},
    ])
    lookup = _make_lookup({"John 3:17": ["world", "save"]})

    valid = _validate_tag_result(tag_result, lookup)

    assert len(valid) == 1
    assert valid[0].tag == tag


def test_invalid_tag_rejected_by_schema() -> None:
    """Pydantic Literal enforcement — invalid tag raises ValidationError at construction."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TaggedCandidate(
            candidate_ref="John 3:17",
            tag="theological_meaning",  # not in EvidenceTag
            justification_terms=["world"],
        )

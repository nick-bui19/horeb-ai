"""
Deterministic TF-IDF similarity scoring over the local ASV corpus.

No external dependencies, no network calls, no LLM scoring.
The LLM is used only downstream (in find_similar) to extract verbatim quotes
from the top-N candidates returned by this module.

Public API:
    score_similarity(seed, scope_book) -> list[CandidateMatch]

CandidateMatch carries the reference, similarity score, and matched terms.
The score is deterministic: same seed + same scope always returns the same ranking.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import pythonbible as pb

from horeb.bible_text import _count_chapter_verses, _get_verse_text
from horeb.schemas import PassageData

# ---------------------------------------------------------------------------
# Stopword list — common English function words that add no signal
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "but", "or", "nor", "for", "yet", "so",
    "of", "in", "on", "at", "to", "by", "up", "as", "is", "it",
    "be", "do", "he", "his", "him", "her", "she", "we", "us", "our",
    "they", "them", "their", "that", "this", "with", "from", "not",
    "was", "are", "were", "had", "has", "have", "been", "into", "unto",
    "thou", "thee", "thy", "thine", "ye", "hath", "doth", "shall",
    "which", "who", "whom", "what", "when", "where", "there", "here",
    "will", "would", "could", "should", "may", "might", "upon", "also",
    "said", "saith", "then", "even", "all", "no", "more", "out",
    "came", "went", "come", "go", "made", "make", "take", "took",
    "i", "me", "my", "mine", "you", "your", "yours", "if", "now",
})

# Minimum token length to consider (single letters are noise)
_MIN_TOKEN_LEN: int = 3

# Maximum candidates to return
DEFAULT_TOP_N: int = 10


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

@dataclass
class _VerseDoc:
    """A single verse represented as a term-frequency vector."""
    ref: str               # "chapter:verse" label
    book_value: int
    chapter: int
    verse: int
    term_freqs: dict[str, float] = field(default_factory=dict)


@dataclass
class CandidateMatch:
    """One candidate similar passage returned by score_similarity."""
    reference: str          # e.g. "John 3:16"
    text: str               # raw verse text with [chapter:verse] label
    similarity_score: float # TF-IDF cosine similarity (0.0–1.0)
    overlap_terms: list[str]  # matched terms in descending IDF weight order


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords and short tokens."""
    raw = re.sub(r"[^a-z\s]", "", text.lower()).split()
    return [t for t in raw if len(t) >= _MIN_TOKEN_LEN and t not in _STOPWORDS]


def _term_freq(tokens: list[str]) -> dict[str, float]:
    """Compute normalised term frequency."""
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    return {t: c / total for t, c in counts.items()}


# ---------------------------------------------------------------------------
# Corpus building
# ---------------------------------------------------------------------------

def _build_corpus(book: pb.Book) -> list[_VerseDoc]:
    """Build a TF-IDF corpus of all verses in the given book."""
    num_chapters = pb.get_number_of_chapters(book)
    docs: list[_VerseDoc] = []
    for chapter in range(1, num_chapters + 1):
        num_verses = _count_chapter_verses(book, chapter)
        for verse in range(1, num_verses + 1):
            text = _get_verse_text(book.value, chapter, verse)
            if text is None:
                continue
            tokens = _tokenise(text)
            docs.append(_VerseDoc(
                ref=f"{chapter}:{verse}",
                book_value=book.value,
                chapter=chapter,
                verse=verse,
                term_freqs=_term_freq(tokens),
            ))
    return docs


def _compute_idf(corpus: list[_VerseDoc]) -> dict[str, float]:
    """Compute IDF for every term in the corpus."""
    n = len(corpus)
    if n == 0:
        return {}
    df: dict[str, int] = {}
    for doc in corpus:
        for term in doc.term_freqs:
            df[term] = df.get(term, 0) + 1
    return {term: math.log(n / count) for term, count in df.items()}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _tfidf_vector(doc: _VerseDoc, idf: dict[str, float]) -> dict[str, float]:
    """Compute the TF-IDF vector for a document."""
    return {
        term: tf * idf.get(term, 0.0)
        for term, tf in doc.term_freqs.items()
    }


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _ranked_overlap_terms(
    seed_vec: dict[str, float],
    candidate_vec: dict[str, float],
    idf: dict[str, float],
    top_n: int = 5,
) -> list[str]:
    """Return overlapping terms sorted by IDF weight descending."""
    common = set(seed_vec) & set(candidate_vec)
    ranked = sorted(common, key=lambda t: idf.get(t, 0.0), reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_similarity(
    seed: PassageData,
    scope_book: pb.Book | None = None,
    top_n: int = DEFAULT_TOP_N,
) -> list[CandidateMatch]:
    """
    Score all verses in scope_book (or the seed's own book if None) by
    TF-IDF cosine similarity against the seed passage.

    Args:
        seed:       The seed passage (retrieved PassageData).
        scope_book: The book to search. Defaults to the seed's own book.
        top_n:      Maximum number of candidates to return.

    Returns:
        List of CandidateMatch sorted by similarity_score descending.
        Does not include verses that overlap with the seed passage itself.
    """
    book = scope_book if scope_book is not None else pb.Book(seed.book)
    book_name = book.name.replace("_", " ").title()

    corpus = _build_corpus(book)
    if not corpus:
        return []

    idf = _compute_idf(corpus)

    # Build seed vector from all tokens in the seed passage text
    seed_tokens = _tokenise(seed.text)
    seed_tf = _term_freq(seed_tokens)
    seed_vec = {term: tf * idf.get(term, 0.0) for term, tf in seed_tf.items()}

    if not seed_vec:
        return []

    # Determine seed verse ID range to exclude from candidates
    seed_verse_ids: set[tuple[int, int]] = {
        (ch, v)
        for ch in range(seed.start_chapter, seed.end_chapter + 1)
        for v in range(
            seed.start_verse if ch == seed.start_chapter else 1,
            (seed.end_verse if ch == seed.end_chapter else _count_chapter_verses(book, ch)) + 1,
        )
    }

    scored: list[tuple[float, _VerseDoc, list[str]]] = []

    for doc in corpus:
        # Skip seed verses
        if (doc.chapter, doc.verse) in seed_verse_ids:
            continue

        candidate_vec = _tfidf_vector(doc, idf)
        score = _cosine_similarity(seed_vec, candidate_vec)

        if score > 0.0:
            overlap = _ranked_overlap_terms(seed_vec, candidate_vec, idf)
            scored.append((score, doc, overlap))

    # Sort by score descending, deterministic tie-break by ref
    scored.sort(key=lambda x: (-x[0], x[1].ref))

    results: list[CandidateMatch] = []
    for score, doc, overlap in scored[:top_n]:
        text = _get_verse_text(doc.book_value, doc.chapter, doc.verse) or ""
        labelled = f"[{doc.chapter}:{doc.verse}] {text}"
        full_ref = f"{book_name} {doc.chapter}:{doc.verse}"
        results.append(CandidateMatch(
            reference=full_ref,
            text=labelled,
            similarity_score=round(score, 6),
            overlap_terms=overlap,
        ))

    return results

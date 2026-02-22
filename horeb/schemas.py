from collections import Counter
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, model_validator


@dataclass
class PassageData:
    reference: str
    book: int               # pythonbible Book enum integer value
    start_chapter: int
    start_verse: int
    end_chapter: int
    end_verse: int
    text: str               # passage text with [chapter:verse] labels
    context_before: str | None  # up to CONTEXT_VERSES_BEFORE preceding verses
    context_after: str | None   # up to CONTEXT_VERSES_AFTER following verses


class QuestionType(str, Enum):
    COMPREHENSION = "comprehension"
    REFLECTION = "reflection"
    APPLICATION = "application"


class Entity(BaseModel):
    name: str
    type: str
    verse_reference: str | None = None
    description: str | None = None


class Question(BaseModel):
    type: QuestionType
    text: str
    verse_reference: str | None = None


class AnalysisResult(BaseModel):
    summary: list[str]
    key_themes: list[str] | None = None
    named_entities: list[Entity] | None = None
    questions: list[Question]
    low_confidence_fields: list[str] = []

    @model_validator(mode="after")
    def validate_summary_length(self) -> "AnalysisResult":
        if len(self.summary) != 3:
            raise ValueError(
                f"summary must have exactly 3 items, got {len(self.summary)}"
            )
        return self

    @model_validator(mode="after")
    def validate_question_distribution(self) -> "AnalysisResult":
        counts = Counter(q.type for q in self.questions)
        expected = {
            QuestionType.COMPREHENSION: 2,
            QuestionType.REFLECTION: 2,
            QuestionType.APPLICATION: 1,
        }
        for qtype, expected_count in expected.items():
            if counts[qtype] != expected_count:
                raise ValueError(
                    f"questions must have exactly {expected_count} "
                    f"{qtype.value} questions, got {counts.get(qtype, 0)}"
                )
        return self

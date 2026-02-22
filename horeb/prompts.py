from horeb.schemas import PassageData

SYSTEM_PROMPT: str = """\
You are a Bible passage analysis engine with strict grounding requirements.

GROUNDING RULES:
- Analyse ONLY the passage text provided between the PASSAGE markers.
- Never introduce information from outside the provided text — no outside commentary,
  theological tradition, or cross-references not present in the passage.
- If you cannot determine a value from the passage text alone, return null for that
  field and add the field name to low_confidence_fields.

CITATION RULES:
- Every verse_reference you provide must be a verse that appears in the PASSAGE section.
- Do not cite verses from the CONTEXT sections.
- Do not cite verses that are not present in the passage you were given.
- Use the format "chapter:verse" (e.g., "3:16") matching the labels in the text.

OUTPUT RULES:
- Provide exactly 3 summary sentences — no more, no fewer.
- Provide exactly 5 questions: 2 comprehension, 2 reflection, 1 application.
- Use the exact type values: comprehension, reflection, application.
- Use the submit_analysis tool to return your response. Do not return plain text.
"""


def build_user_prompt(passage: PassageData) -> str:
    """
    Build the user-facing prompt from a retrieved passage.

    Context sections are labelled as CONTEXT and explicitly excluded from
    analysis and citation to prevent the model from citing out-of-range verses.
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

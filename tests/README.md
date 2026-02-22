# Test Matrix

Every failure path in Horeb must have a corresponding test. This file is the canonical record.
No failure path ships without coverage.

| # | Path | Condition | Test file | Test name |
|---|------|-----------|-----------|-----------|
| 1 | Valid reference parsing | John 3:16-21 | test_bible_text.py | TestValidReferences::test_john_3_16_21 |
| 1 | Valid reference parsing | Psalm 23:1-6 | test_bible_text.py | TestValidReferences::test_psalm_23_1_6 |
| 1 | Valid reference parsing | Single verse | test_bible_text.py | TestValidReferences::test_single_verse |
| 2 | Invalid reference → InvalidReferenceError | Empty string | test_bible_text.py | TestInvalidReferences::test_empty_string |
| 2 | Invalid reference → InvalidReferenceError | Whitespace only | test_bible_text.py | TestInvalidReferences::test_whitespace_only |
| 2 | Invalid reference → InvalidReferenceError | Chapter without verses | test_bible_text.py | TestInvalidReferences::test_chapter_without_verses |
| 2 | Invalid reference → InvalidReferenceError | Malformed verse numbers | test_bible_text.py | TestInvalidReferences::test_malformed_verse_numbers |
| 2 | Invalid reference → InvalidReferenceError | Not a reference | test_bible_text.py | TestInvalidReferences::test_not_a_bible_reference |
| 3 | Passage too long → InvalidReferenceError | Exceeds MAX_PASSAGE_VERSES | test_bible_text.py | TestPassageLengthLimit::test_exceeds_max_raises_invalid_reference |
| 3 | Passage too long → InvalidReferenceError | Exactly at max | test_bible_text.py | TestPassageLengthLimit::test_exactly_max_verses_accepted |
| 3 | Passage too long → InvalidReferenceError | One over max | test_bible_text.py | TestPassageLengthLimit::test_one_over_max_rejected |
| 4 | EmptyPassageError | Retrieved text too short | test_engine.py | TestEmptyPassageGuard::test_empty_passage_text_raises_before_llm_call |
| 5 | Context clamped at book start | Genesis 1:1 → context_before=None | test_bible_text.py | TestContextBoundaryClamping::test_genesis_1_1_has_no_context_before |
| 6 | Context clamped at book end | Revelation 22:21 → context_after=None | test_bible_text.py | TestContextBoundaryClamping::test_revelation_22_21_has_no_context_after |
| 7 | Single-chapter book | Jude 3 | test_bible_text.py | TestSingleChapterBooks::test_jude_3 |
| 7 | Single-chapter book | Philemon 1:10 | test_bible_text.py | TestSingleChapterBooks::test_philemon_1_10 |
| 7 | Single-chapter book | Obadiah 1:3 | test_bible_text.py | TestSingleChapterBooks::test_obadiah_1_3 |
| 8 | Valid LLM response parses cleanly | Stage 1 direct parse | test_repair.py | TestStage1DirectParse::test_valid_response_succeeds_without_repair |
| 9 | Malformed JSON → json_repair → success | Truncated JSON | test_repair.py | TestStage2JsonRepair::test_truncated_json_repaired_to_valid_result |
| 10 | Schema failure → LLM retry → success | Wrong distribution, valid retry | test_repair.py | TestStage3LLMRetry::test_retry_succeeds_on_schema_failure |
| 11 | All retry paths fail → AnalysisFailedError | Both calls return invalid | test_repair.py | TestAllStagesFail::test_raises_analysis_failed_error |
| 11 | raw_response preserved in error | Bad retry response | test_repair.py | TestAllStagesFail::test_raw_response_preserved_in_error |
| 11 | Hard ceiling of 2 LLM calls | Never more than 1 retry | test_repair.py | TestAllStagesFail::test_max_2_total_llm_calls |
| 12 | Wrong summary length → ValidationError | 2 items | test_schemas.py | TestSummaryLengthValidator::test_2_items_rejected |
| 12 | Wrong summary length → ValidationError | 4 items | test_schemas.py | TestSummaryLengthValidator::test_4_items_rejected |
| 13 | Wrong question distribution → ValidationError | 3 comprehension | test_schemas.py | TestQuestionDistributionValidator::test_3_comprehension_rejected |
| 13 | Wrong question distribution → ValidationError | 0 application | test_schemas.py | TestQuestionDistributionValidator::test_0_application_rejected |
| 14 | Citation in range → passes | John 3:16-21 cites in range | test_engine.py | TestCitationVerification::test_in_range_citation_passes |
| 15 | Citation out of range → CitationOutOfRangeError | John 3:22 outside 3:16-21 | test_engine.py | TestCitationVerification::test_out_of_range_citation_raises |
| 16 | Null key_themes accepted | key_themes=None | test_schemas.py | TestNullableFields::test_key_themes_none_accepted |
| 16 | Null named_entities accepted | named_entities=None | test_schemas.py | TestNullableFields::test_named_entities_none_accepted |
| 17 | CLI exit code 2 | InvalidReferenceError | test_cli.py | TestExitCodes::test_invalid_reference_exit_code_2 |
| 18 | CLI exit code 3 | EmptyPassageError | test_cli.py | TestExitCodes::test_empty_passage_exit_code_3 |
| 19 | CLI exit code 4 | CitationOutOfRangeError | test_cli.py | TestExitCodes::test_citation_out_of_range_exit_code_4 |
| 20 | CLI exit code 5 | AnalysisFailedError | test_cli.py | TestExitCodes::test_analysis_failed_exit_code_5 |
| 20 | [WARN] logged on retry | stderr output | test_repair.py | TestStage3LLMRetry::test_warn_logged_to_stderr_on_retry |
| 21 | Snapshot: John 3:16-21 result structure | Full AnalysisResult shape | test_snapshots.py | TestJohnSnapshot::* |

## Running tests

```bash
# All unit tests (no API key needed)
uv run pytest

# Exclude integration tests explicitly
uv run pytest -m "not integration"

# Run only integration tests (requires ANTHROPIC_API_KEY)
uv run pytest -m integration

# Run with verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_schemas.py -v
```

## Adding new paths

When a new error type is added to `errors.py`, add a row to this table and
a corresponding test before the PR is merged. The matrix is the contract.

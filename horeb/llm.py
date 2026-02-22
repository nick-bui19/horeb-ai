import json
import os
import sys
from typing import Protocol

import anthropic

from horeb.errors import AnalysisFailedError
from horeb.schemas import AnalysisResult

_MODEL = "claude-haiku-4-5-20251001"


class LLMProvider(Protocol):
    def complete(self, system: str, prompt: str) -> str:
        """Send system + user prompt to the LLM and return raw response string."""
        ...


def _build_analysis_tool() -> dict:
    """
    Build the Claude tool definition from the AnalysisResult JSON Schema.

    Using tool use (structured output) rather than prompt-embedded schema:
    - Removes schema tokens from the user prompt
    - Claude's API enforces JSON structure at the transport level
    - Reduces repair/retry frequency for structural issues
    """
    schema = AnalysisResult.model_json_schema()
    schema.pop("title", None)  # avoid conflict with tool-level "name" field
    return {
        "name": "submit_analysis",
        "description": "Submit the structured Bible passage analysis.",
        "input_schema": schema,
    }


_ANALYSIS_TOOL = _build_analysis_tool()


class ClaudeProvider:
    """
    LLMProvider implementation backed by the Anthropic API.

    Uses forced tool use to obtain structured AnalysisResult output.
    Set HOREB_DEBUG=1 to log estimated prompt token counts before each call.
    """

    def complete(self, system: str, prompt: str) -> str:
        if os.environ.get("HOREB_DEBUG") == "1":
            total_chars = len(system) + len(prompt)
            estimated_tokens = total_chars // 4
            print(
                f"[DEBUG] Prompt chars: {total_chars}, "
                f"estimated tokens: ~{estimated_tokens}",
                file=sys.stderr,
            )

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=_MODEL,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            tools=[_ANALYSIS_TOOL],
            tool_choice={"type": "any"},
        )

        for block in response.content:
            if block.type == "tool_use":
                # block.input is already a parsed dict; serialise to string
                # so LLMProvider.complete() always returns str (testable contract)
                return json.dumps(block.input)

        # tool_choice="any" guarantees a tool call — this path should not be reached
        raise AnalysisFailedError(
            "LLM response contained no tool call",
            raw_response=str(response.content),
        )

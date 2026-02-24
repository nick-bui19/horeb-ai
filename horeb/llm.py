import json
import os
import sys
from typing import Protocol

import anthropic
from pydantic import BaseModel

from horeb.errors import AnalysisFailedError

_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_MAX_TOKENS = 2048


class LLMProvider(Protocol):
    def complete(
        self,
        system: str,
        prompt: str,
        schema: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Send system + user prompt to the LLM and return raw response string.

        Args:
            system:     System prompt string.
            prompt:     User prompt string.
            schema:     Pydantic model class to build the tool schema from.
                        If None, no tool is forced (plain text response).
            max_tokens: Override the default token limit for this call.
                        Use higher values (e.g. 4096) for synthesis responses.
        """
        ...


def _build_tool_for_schema(schema: type[BaseModel], tool_name: str = "submit_result") -> dict:
    """
    Build a Claude tool definition from any Pydantic model's JSON Schema.

    Using tool use (structured output) rather than prompt-embedded schema:
    - Removes schema tokens from the user prompt
    - Claude's API enforces JSON structure at the transport level
    - Reduces repair/retry frequency for structural issues
    """
    json_schema = schema.model_json_schema()
    json_schema.pop("title", None)  # avoid conflict with tool-level "name" field
    return {
        "name": tool_name,
        "description": f"Submit the structured {schema.__name__} result.",
        "input_schema": json_schema,
    }


class ClaudeProvider:
    """
    LLMProvider implementation backed by the Anthropic API.

    The Anthropic client is created once at construction time and reused
    across all calls — avoids re-reading env vars and re-allocating HTTP
    connection pools on every segment call.

    Set HOREB_DEBUG=1 to log estimated prompt token counts before each call.
    """

    def __init__(self, max_tokens: int = _DEFAULT_MAX_TOKENS) -> None:
        self._client = anthropic.Anthropic()
        self._default_max_tokens = max_tokens

    def complete(
        self,
        system: str,
        prompt: str,
        schema: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        tokens = max_tokens if max_tokens is not None else self._default_max_tokens

        if os.environ.get("HOREB_DEBUG") == "1":
            total_chars = len(system) + len(prompt)
            estimated_tokens = total_chars // 4
            print(
                f"[DEBUG] Prompt chars: {total_chars}, "
                f"estimated tokens: ~{estimated_tokens}, "
                f"max_tokens: {tokens}",
                file=sys.stderr,
            )

        kwargs: dict = dict(
            model=_MODEL,
            max_tokens=tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        if schema is not None:
            tool = _build_tool_for_schema(schema)
            kwargs["tools"] = [tool]
            kwargs["tool_choice"] = {"type": "any"}

        response = self._client.messages.create(**kwargs)

        # If a tool was forced, extract the tool call input
        if schema is not None:
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

        # Plain text response (no tool)
        for block in response.content:
            if hasattr(block, "text"):
                return block.text

        raise AnalysisFailedError(
            "LLM response contained no text content",
            raw_response=str(response.content),
        )

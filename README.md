# Horeb

Horeb is a CLI engine Bible passage analysis with zero hallucinations. It can generate summaries, themes, citations, book outlines, and similarity search, all anchored strictly to the text with no theological commentary added.

Built with Python 3.11, pythonbible (ASV), Pydantic v2, and Typer.

## Try it yourself

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/), an Anthropic API key.

```sh
git clone https://github.com/nickbui/horeb.git
cd horeb
uv sync
cp .env.example .env  # add your ANTHROPIC_API_KEY
```

**Analyze a passage, chapter, or whole book:**
```sh
uv run horeb analyze "John 3:16-21"
uv run horeb analyze "Romans 8"
uv run horeb analyze "Ruth" --output ruth.md
```

**Find similar passages (TF-IDF, no LLM):**
```sh
uv run horeb find-similar "Psalm 23:1-6"
uv run horeb find-similar "Psalm 23:1-6" --book Psalms --top-n 5 --tags
```

**Run the tests:**
```sh
uv run pytest -m "not integration"   # 232 unit tests, no API key needed
uv run pytest -m integration         # live API calls, requires ANTHROPIC_API_KEY
```

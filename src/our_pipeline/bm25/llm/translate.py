"""LLM-based query translation for multilingual BM25 search."""

from __future__ import annotations

import logging
import re

from our_pipeline.constants import CONFIG
from our_pipeline.bm25.llm.load_llm import llm

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "You are a legal translation assistant for Swiss law "
    "(Bundesrecht / droit fédéral / diritto federale).\n"
    "Translate the following query ONLY — do not answer it.\n"
    "Respond with exactly four lines and nothing else:\n"
    "EN: <English translation>\n"
    "DE: <German translation>\n"
    "FR: <French translation>\n"
    "IT: <Italian translation>\n\n"
    "Query: {query}"
)


def translate_query(query: str) -> dict[str, str]:
    """Translate a legal query to English, German, and French using the shared LLM.

    Uses a structured prompt requesting exactly three labelled lines.
    Returns only the languages that were successfully parsed; returns an
    empty dict on complete failure so callers can fall back to the original.

    Args:
        query: Query string in any language.

    Returns:
        Dict with zero to three entries: ``{"en": "...", "de": "...", "fr": "..."}``.
    """
    prompt = _PROMPT_TEMPLATE.format(query=query)
    try:
        raw = llm(
            prompt,
            max_tokens=512,
            temperature=0.0,
            model=CONFIG["translation_model"],
        )["choices"][0]["text"]
    except Exception as exc:
        logger.warning("Translation LLM call failed: %s", exc)
        return {}

    # Strip any inline reasoning block in case a thinking-capable model is used.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    translations: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("EN:"):
            text = line[3:].strip()
            if text:
                translations["en"] = text
        elif line.upper().startswith("DE:"):
            text = line[3:].strip()
            if text:
                translations["de"] = text
        elif line.upper().startswith("FR:"):
            text = line[3:].strip()
            if text:
                translations["fr"] = text
        elif line.upper().startswith("IT:"):
            text = line[3:].strip()
            if text:
                translations["it"] = text

    if not translations:
        logger.warning("Translation parsing failed for query %r; raw=%r", query, raw[:200])
    return translations

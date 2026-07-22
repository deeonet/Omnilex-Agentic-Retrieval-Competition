"""LLM-based query decomposition for targeted multi-issue retrieval.

Swiss legal queries (especially the val/test hypotheticals) raise several
distinct legal issues and cite many provisions. A single broad search cannot
surface them all. ``decompose_query`` breaks a query into focused sub-issues,
each with precise German legal search terms (Fachbegriffe) that go straight to
BM25 — sidestepping the "English query → German corpus" lexical mismatch.
"""

from __future__ import annotations

import json
import logging
import re

from constants import CONFIG
from llm.load_llm import llm

logger = logging.getLogger(__name__)

# Cache so re-running on the same query (e.g. debugging) costs only one call.
_DECOMPOSE_CACHE: dict[str, list[dict]] = {}

_PROMPT_TEMPLATE = (
    "You are a Swiss legal research assistant. A user asks a legal question that "
    "may raise several distinct legal issues, each requiring different statutory "
    "provisions or court decisions.\n\n"
    "Break the question into its distinct legal issues. For EACH issue, output an "
    "object with exactly these fields:\n"
    '  "issue": a short description of the legal issue (English)\n'
    '  "de_keywords": precise German legal search terms (Fachbegriffe) for '
    "retrieving the relevant statute or case text — key terms only, NOT a full "
    "sentence\n"
    '  "type": one of "law", "court", or "both" (does the issue need statute law, '
    "case law, or both)\n\n"
    "Rules:\n"
    "- Output ONLY a JSON array of objects. No prose, no markdown fences.\n"
    "- Use real Swiss legal terminology in de_keywords (e.g. 'Untersuchungshaft "
    "Kollusionsgefahr', 'Fahrlässigkeit Sorgfaltspflicht').\n"
    "- At most {max_issues} issues.\n\n"
    "Question: {query}"
)


def _extract_json_array(raw: str) -> list[dict]:
    """Pull the first JSON array out of a model response, defensively."""
    # Drop any <think> reasoning block from thinking-capable models.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Strip markdown code fences if present.
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()

    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    issues: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        kw = str(item.get("de_keywords", "")).strip()
        issue = str(item.get("issue", "")).strip()
        if not kw and not issue:
            continue
        itype = str(item.get("type", "both")).strip().lower()
        if itype not in ("law", "court", "both"):
            itype = "both"
        issues.append({"issue": issue, "de_keywords": kw or issue, "type": itype})
    return issues


def decompose_query(query: str) -> list[dict]:
    """Decompose a legal query into focused sub-issues with German search terms.

    Args:
        query: The natural-language legal query (any language).

    Returns:
        List of ``{"issue", "de_keywords", "type"}`` dicts. Empty list on failure
        (the caller is expected to fall back to a single broad search).
    """
    if query in _DECOMPOSE_CACHE:
        return _DECOMPOSE_CACHE[query]

    max_issues = CONFIG.get("max_decompose_issues", 12)
    prompt = _PROMPT_TEMPLATE.format(max_issues=max_issues, query=query)
    try:
        raw = llm(
            prompt,
            max_tokens=CONFIG.get("decompose_max_tokens", 1024),
            temperature=0.0,
            model=CONFIG.get("decompose_model", CONFIG.get("translation_model")),
        )["choices"][0]["text"]
    except Exception as exc:  # noqa: BLE001 — never let decomposition kill a query
        logger.warning("Decomposition LLM call failed: %s", exc)
        return []

    issues = _extract_json_array(raw)[:max_issues]
    if not issues:
        logger.warning("Decomposition parsing failed for query %r; raw=%r", query, raw[:200])

    _DECOMPOSE_CACHE[query] = issues
    return issues

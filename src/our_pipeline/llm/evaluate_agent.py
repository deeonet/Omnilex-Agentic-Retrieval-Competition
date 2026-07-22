"""Evaluation agent — LQ-RAG's RAG-triad, adapted to citation-set selection.

The retrieval stage (``define_agent.run_agent``) maximises candidate-pool *recall*
and applies no relevance selection. This module is the missing *precision* stage:
an LLM judge scores each candidate citation for **groundedness + relevance** against
its retrieved text and drops false positives — the Macro-F1 precision lever.

It also exposes ``reformulate`` for the recursive-feedback loop (LQ-RAG Algorithm 3):
when a query yields too few grounded citations, the query is rephrased with German
legal terminology and retrieval is re-run by the caller.

Judgements are batched (many candidates per LLM call) to keep latency bounded, as
LQ-RAG notes the eval agent roughly doubles response time.
"""
from __future__ import annotations

import json
import logging
import re

from constants import CONFIG
from llm.load_llm import llm

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = (
    "You are a Swiss legal expert deciding which citations are RELEVANT to a legal "
    "question. A citation is relevant only if its text directly addresses a legal "
    "issue raised by the question (groundedness). Be strict — drop citations that are "
    "merely topically adjacent or generic.\n\n"
    "Question:\n{query}\n\n"
    "Candidate citations (index: citation — text excerpt):\n{candidates}\n\n"
    "Return ONLY a JSON array of the indices (integers) of the RELEVANT citations. "
    "No prose, no markdown. Example: [0, 3, 4]"
)

_REFORMULATE_PROMPT = (
    "You are a Swiss legal research assistant. The following legal issue returned no "
    "clearly relevant statute or case-law citations. Rewrite it as a focused search "
    "query using precise German legal terminology (Fachbegriffe) that would retrieve "
    "the governing provision or leading decision. Return ONLY the German search terms, "
    "no prose.\n\nIssue: {issue}"
)


def _extract_int_array(raw: str) -> list[int]:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        # Fall back to grabbing bare integers.
        return [int(x) for x in re.findall(r"\d+", raw[start : end + 1])]
    return [int(x) for x in parsed if isinstance(x, (int, float))]


class EvaluationAgent:
    """Filters a recall-maximised candidate pool down to grounded, relevant citations."""

    def __init__(
        self,
        cite2text: dict[str, str],
        batch_size: int | None = None,
        max_candidates: int | None = None,
        excerpt_chars: int | None = None,
    ) -> None:
        self.cite2text = cite2text
        self.batch_size = batch_size or CONFIG.get("eval_batch_size", 10)
        self.max_candidates = max_candidates or CONFIG.get("eval_max_candidates", 80)
        self.excerpt_chars = excerpt_chars or CONFIG.get("eval_excerpt_chars", 400)

    def _judge_batch(self, query: str, batch: list[str]) -> list[str]:
        """Return the subset of ``batch`` the LLM judges relevant."""
        lines = []
        for i, cit in enumerate(batch):
            txt = (self.cite2text.get(cit, "") or "")[: self.excerpt_chars]
            lines.append(f"{i}: {cit} — {txt}")
        prompt = _JUDGE_PROMPT.format(query=query, candidates="\n".join(lines))
        try:
            raw = llm(
                prompt,
                max_tokens=CONFIG.get("eval_max_tokens", 256),
                temperature=0.0,
                model=CONFIG.get("eval_model", CONFIG.get("decompose_model")),
            )["choices"][0]["text"]
        except Exception as exc:  # noqa: BLE001 — never let judging kill a query
            logger.warning("Judge LLM call failed: %s", exc)
            return batch  # fail open: keep the batch rather than lose recall
        keep_idx = {i for i in _extract_int_array(raw) if 0 <= i < len(batch)}
        return [batch[i] for i in keep_idx]

    def filter(self, query: str, candidates: list[str]) -> tuple[list[str], dict]:
        """Keep only grounded/relevant citations, preserving input order.

        Never returns empty when candidates exist: if the judge rejects everything,
        falls back to the top candidates to protect recall.
        """
        if not candidates:
            return [], {"type": "eval_agent", "in": 0, "kept": 0}

        pool = candidates[: self.max_candidates]
        kept: list[str] = []
        for start in range(0, len(pool), self.batch_size):
            batch = pool[start : start + self.batch_size]
            kept.extend(self._judge_batch(query, batch))

        # Preserve original order.
        kept_set = set(kept)
        kept_ordered = [c for c in pool if c in kept_set]

        fallback = False
        if not kept_ordered:
            fallback = True
            kept_ordered = pool[: CONFIG.get("eval_fallback_top_k", 10)]

        log = {
            "type": "eval_agent",
            "in": len(candidates),
            "judged": len(pool),
            "kept": len(kept_ordered),
            "fallback": fallback,
        }
        return kept_ordered, log


def reformulate(issue: str) -> str:
    """Rephrase an under-performing issue into German legal search terms (feedback loop)."""
    prompt = _REFORMULATE_PROMPT.format(issue=issue)
    try:
        raw = llm(
            prompt,
            max_tokens=128,
            temperature=0.2,
            model=CONFIG.get("eval_model", CONFIG.get("decompose_model")),
        )["choices"][0]["text"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reformulate LLM call failed: %s", exc)
        return issue
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or issue

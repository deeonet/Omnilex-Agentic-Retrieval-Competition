"""Sibling-consideration expansion for court-decision candidates.

Court citations carry a specific consideration, e.g. ``BGE 137 IV 122 E. 6.2``
or ``1B_28/2022 E. 4.1``. Retrieval routinely surfaces the *right decision* but
the *wrong consideration* (it found ``BGE 137 IV 122 E. 4.2`` while the gold
wants ``E. 4.1`` / ``E. 6.2`` / ``E. 6.4``). Gold answers also frequently cite
several considerations of the same leading decision.

This module builds a decision -> all-its-considerations index from the court
corpus, so that once any consideration of a decision is retrieved we can add the
decision's sibling considerations to the candidate pool — a cheap recall lift
that turns decision-level hits into exact-consideration hits.
"""

from __future__ import annotations

# Trailing "E. <consideration>" (or "E <consideration>") marker, e.g. " E. 6.2",
# " E 2.", " E. 1.12.2011" (date-mangled). Stripping it yields the decision identity.
import re

_CONSIDERATION_RE = re.compile(r"\s*E\.?\s*[\dA-Za-z.]+\s*$")


def is_court_citation(citation: str) -> bool:
    """True for BGE / docket-style court citations carrying a consideration."""
    c = citation.strip()
    has_consideration = " E." in c or " E " in c
    return has_consideration and (c.startswith("BGE") or "/" in c)


def court_decision_key(citation: str) -> str:
    """Strip the consideration tail to get the decision identity.

    ``BGE 137 IV 122 E. 6.2`` -> ``BGE 137 IV 122``
    ``1B_28/2022 E. 4.1``     -> ``1B_28/2022``
    """
    return _CONSIDERATION_RE.sub("", citation.strip()).strip()


def build_court_sibling_index(
    documents: list[dict], citation_field: str = "citation"
) -> dict[str, list[str]]:
    """Map decision key -> sorted list of every court citation for that decision.

    Args:
        documents: Court corpus docs (each a dict with a citation field).
        citation_field: Key holding the citation string.

    Returns:
        ``{decision_key: [citation, ...]}`` for all court decisions in the corpus.
    """
    grouped: dict[str, set[str]] = {}
    for doc in documents:
        citation = doc.get(citation_field)
        if not citation or not is_court_citation(citation):
            continue
        key = court_decision_key(citation)
        if key:
            grouped.setdefault(key, set()).add(citation)
    return {key: sorted(values) for key, values in grouped.items()}


def expand_court_siblings(
    citations: list[str],
    sibling_index: dict[str, list[str]],
    max_per_decision: int | None = None,
) -> tuple[list[str], int]:
    """Add sibling considerations for every retrieved court decision.

    Order-preserving: each original citation is followed by its (not-yet-seen)
    siblings. Non-court citations pass through untouched.

    Args:
        citations: Retrieved candidate citations (order preserved).
        sibling_index: Output of :func:`build_court_sibling_index`.
        max_per_decision: Cap on siblings added per decision (None = all).

    Returns:
        ``(expanded_citations, num_added)``.
    """
    expanded: list[str] = []
    seen: set[str] = set()
    added = 0

    def _push(c: str) -> bool:
        if c in seen:
            return False
        seen.add(c)
        expanded.append(c)
        return True

    for citation in citations:
        _push(citation)
        if not is_court_citation(citation):
            continue
        siblings = sibling_index.get(court_decision_key(citation), [])
        if max_per_decision is not None:
            siblings = siblings[:max_per_decision]
        for sibling in siblings:
            if _push(sibling):
                added += 1

    return expanded, added

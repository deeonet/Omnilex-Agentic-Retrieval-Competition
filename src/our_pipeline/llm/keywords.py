"""Query -> German keyword / explicit-citation extraction for BM25 law search.

The law corpus (``laws_de.csv``) is German-only, so the most effective BM25 query is a
concise set of German legal terms rather than a translated full-sentence question. We also
pull out any statute articles the query names outright (e.g. "Art. 221 Abs. 1 lit. b StPO")
as high-precision search variants.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from omnilex.citations import CitationNormalizer
from omnilex.citations.abbreviations import get_german_abbreviations, load_abbreviations
from our_pipeline.constants import CONFIG
from our_pipeline.llm.load_llm import llm

logger = logging.getLogger(__name__)

_NORMALIZER = CitationNormalizer()

_KEYWORD_PROMPT = (
    "You are a retrieval assistant for Swiss federal law (Bundesrecht).\n"
    "Read the legal question and output GERMAN keywords for a keyword (BM25) search "
    "over German statute texts.\n"
    "Rules:\n"
    "- Output GERMAN only, a single line of space-separated terms.\n"
    "- At most 15 DISTINCT terms. Never repeat a term.\n"
    "- Include the core legal concepts and technical terms in canonical German "
    "(e.g. Untersuchungshaft, Kollusionsgefahr, Verhältnismässigkeit, Beschwerde).\n"
    "- Include relevant statute abbreviations, always the GERMAN form "
    "(IVG not LAI, OR not CO, ZGB not CC, StGB not CP, SchKG not LP).\n"
    "- No sentences, no explanations, no numbering.\n\n"
    "Question: {query}\n"
    "Keywords:"
)

# Safeguard cap on distinct keyword tokens (instruct models occasionally loop).
_MAX_KEYWORD_TOKENS = 24

# Article reference up to (and including) the law-code token that follows it.
_CITATION_REF = re.compile(
    r"(Art(?:ikel)?\.?\s*\d+[a-z]?"                               # Art. 221
    r"(?:\s+Abs\.?\s*\d+[a-z]?)?"                                 # optional Abs. 1
    r"(?:\s+(?:lit\.?|Bst\.?|Ziff\.?|Nr\.?)\s*[a-z0-9]+)?)"       # optional lit. b
    r"\s+([A-Za-zÄÖÜäöüéè][A-Za-zÄÖÜäöüéè.\-]*)",                 # following candidate law code
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _abbrev_lookup() -> dict[str, str]:
    """Lowercase law-abbreviation -> canonical form (e.g. {"stpo": "StPO"})."""
    return {a.lower(): a for a in get_german_abbreviations()}


@lru_cache(maxsize=1)
def _foreign_to_german() -> dict[str, str]:
    """French/Italian law abbreviation -> German equivalent (e.g. {"LAI": "IVG"}).

    The German abbreviation is the strongest BM25 signal (it appears in every
    article's citation and title), so a wrong-language code badly degrades retrieval.
    Built from ``abbrev-translations.json``; only unambiguous remaps are kept.
    """
    german = set(get_german_abbreviations())
    mapping: dict[str, str] = {}
    for entry in load_abbreviations().values():
        de = entry.get("de")
        if not de or de[0].isdigit():
            continue
        for variant in (entry.get("fr"), entry.get("it")):
            if variant and variant != de and variant not in german:
                mapping.setdefault(variant, de)
    return mapping


def extract_legal_keywords(query: str) -> str:
    """Extract a single line of German legal search keywords from a query.

    Args:
        query: Legal question in any language.

    Returns:
        A space-separated German keyword string, or "" if extraction failed.
    """
    prompt = _KEYWORD_PROMPT.format(query=query)
    try:
        raw = llm(
            prompt,
            max_tokens=128,
            temperature=0.0,
            model=CONFIG["translation_model"],
        )["choices"][0]["text"]
    except Exception as exc:
        logger.warning("Keyword extraction LLM call failed: %s", exc)
        return ""

    # Strip any inline reasoning block in case a thinking-capable model is used.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    for line in raw.splitlines():
        line = re.sub(r"^(keywords|stichw[oö]rter)\s*:\s*", "", line.strip(), flags=re.IGNORECASE)
        if line:
            return _dedupe_terms(line)

    logger.warning("Keyword extraction returned no usable line for query %r", query[:120])
    return ""


def _dedupe_terms(line: str) -> str:
    """Remap foreign law codes to German, drop duplicates, and cap the count.

    The foreign->German remap fixes cases where the model emits e.g. "LAI" for "IVG".
    Dedup guards against the instruct model occasionally looping on a term, which would
    otherwise over-weight it in the BM25 score (duplicate query tokens are summed).
    """
    foreign = _foreign_to_german()
    seen: set[str] = set()
    out: list[str] = []
    for term in line.split():
        term = foreign.get(term, term)
        key = term.lower()
        if key not in seen:
            seen.add(key)
            out.append(term)
        if len(out) >= _MAX_KEYWORD_TOKENS:
            break
    return " ".join(out)


def extract_explicit_citations(query: str) -> list[str]:
    """Extract statute references the query names outright, in canonical form.

    Only matches whose trailing token is a known German law abbreviation are kept, so
    free-text mentions of "Art. 5" without a law code are ignored. Each match is run
    through ``CitationNormalizer`` so e.g. "Art. 221 Abs. 1 lit. b StPO" becomes
    "Art. 221 Abs. 1 StPO" — the surface form stored in the corpus — enabling an exact
    ``BM25Index.get_by_citation`` lookup.

    Args:
        query: Legal question in any language.

    Returns:
        Deduplicated list of canonical citation strings.
    """
    out: list[str] = []
    seen: set[str] = set()
    lookup = _abbrev_lookup()
    for ref, code in _CITATION_REF.findall(query):
        canonical_code = lookup.get(code.strip(" .,;:").lower())
        if not canonical_code:
            continue
        ref_norm = re.sub(r"\s+", " ", ref).strip()
        raw = f"{ref_norm} {canonical_code}"
        citation = _NORMALIZER.canonicalize(raw) or raw
        if citation not in seen:
            seen.add(citation)
            out.append(citation)
    return out

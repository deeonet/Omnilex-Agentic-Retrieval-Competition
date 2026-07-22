import re
from constants import CONFIG
from expand import expand_court_siblings
from llm.decompose import decompose_query


def parse_all_agent_actions(response: str) -> list[tuple[str, str]]:
    """
    Parse ALL action/input pairs from agent response.
    
    The LLM may output multiple actions in one response. This function
    extracts all of them.
    
    Args:
        response: Full LLM response text
        
    Returns:
        List of (action, action_input) tuples
    """
    actions = []
    
    # Find all "Action:" lines
    action_pattern = r"Action:\s*(\w+)"
    # Stop the Action Input at a blank line or the next ReAct section keyword so
    # trailing "Thought:" reasoning doesn't leak into the search query.
    input_pattern = (
        r"Action Input:[ \t]*(.+?)"
        r"(?=\n\s*\n|\n\s*(?:Thought|Action|Observation|Final Answer)\s*:|$)"
    )
    
    # Find all action matches with their positions
    action_matches = list(re.finditer(action_pattern, response, re.IGNORECASE))
    
    for i, action_match in enumerate(action_matches):
        action = action_match.group(1).strip()
        
        # Find the corresponding Action Input
        # Start search after the Action line
        start_pos = action_match.end()
        # End search at next Action or end of string
        if i + 1 < len(action_matches):
            end_pos = action_matches[i + 1].start()
        else:
            end_pos = len(response)
        
        input_text = response[start_pos:end_pos]
        input_match = re.search(input_pattern, input_text, re.IGNORECASE | re.DOTALL)
        
        if input_match:
            action_input = input_match.group(1).strip()
            actions.append((action, action_input))
    
    return actions


def extract_citations_from_text(text: str) -> list[str]:
    """Extract citations from any text (tool output or final answer)."""
    citations = []
    
    # SR pattern: SR followed by number (optionally with article)
    sr_matches = re.findall(
        r"SR\s*\d{3}(?:\.\d+)?(?:\s+Art\.?\s*\d+[a-z]?)?",
        text,
        re.IGNORECASE
    )
    citations.extend(sr_matches)
    
    # BGE pattern: BGE volume section page
    bge_matches = re.findall(
        r"BGE\s+\d{1,3}\s+[IVX]+[a-z]?\s+\d+(?:\s+E\.\s*\d+[a-z]?)?",
        text,
        re.IGNORECASE
    )
    citations.extend(bge_matches)
    
    # Art. pattern: Art. X LAW (e.g., Art. 1 ZGB, Art. 41 OR)
    art_matches = re.findall(
        r"Art\.?\s+\d+[a-z]?\s+(?:Abs\.?\s*\d+\s+)?[A-Z]{2,}",
        text,
        re.IGNORECASE
    )
    citations.extend(art_matches)
    
    return list(set(citations))


def truncate_observation_for_llm(observation: str, max_chars: int = 1200) -> str:
    """Truncate observation text for LLM context, preserving data elsewhere.
    
    This truncates only the text sent to the LLM in the conversation.
    Full observations remain in logs and are used for citation extraction.
    
    Args:
        observation: Full observation text
        max_chars: Maximum characters to keep
        
    Returns:
        Truncated observation text
    """
    if len(observation) <= max_chars:
        return observation
    
    # Truncate and add indicator
    return observation[:max_chars] + f"\n... (truncated, {len(observation) - max_chars} chars remaining)"


def truncate_conversation(conversation: str, max_chars: int) -> str:
    """Truncate conversation to fit within token budget, keeping system prompt and recent context.
    
    Args:
        conversation: Full conversation text
        max_chars: Maximum characters to keep
        
    Returns:
        Truncated conversation text
    """
    if len(conversation) <= max_chars:
        return conversation
    
    # Find the system prompt end marker and keep it
    inst_end = conversation.find("[/INST]")
    if inst_end == -1:
        # Fallback: keep last max_chars
        return "..." + conversation[-max_chars:]
    
    system_part = conversation[:inst_end + 7]  # Include [/INST]
    remaining_budget = max_chars - len(system_part) - 100  # Buffer for truncation marker
    
    if remaining_budget <= 0:
        # System prompt itself is too long, just truncate from end
        return conversation[-max_chars:]
    
    # Keep the most recent conversation
    rest = conversation[inst_end + 7:]
    if len(rest) > remaining_budget:
        rest = "\n...[earlier conversation truncated]...\n" + rest[-remaining_budget:]
    
    return system_part + rest


def _tool_targets(issue_type: str | None) -> list[str]:
    """Map a decomposed issue 'type' to the search tools to invoke."""
    t = (issue_type or "both").lower()
    if t == "law":
        return ["search_laws"]
    if t == "court":
        return ["search_courts"]
    return ["search_laws", "search_courts"]


def run_agent(
    query: str,
    tools: dict[str, callable],
    verbose: bool = False,
    sibling_index: dict[str, list[str]] | None = None,
) -> tuple[list[str], list[dict]]:
    """Retrieve citations via deterministic query decomposition.

    Pipeline (no ReAct loop):
      1. Decompose the query into focused legal sub-issues, each with precise
         German search terms (see ``llm.decompose.decompose_query``).
      2. For every sub-issue, run the relevant search tool(s) and collect the
         returned candidate citations into a deduplicated, order-preserving pool.
      3. Optionally expand court candidates to their sibling considerations
         (``sibling_index``; turns decision-level hits into exact-consideration
         hits — see ``expand.expand_court_siblings``).
      4. Return the candidate pool (the union of all retrieved citations).

    Args:
        sibling_index: decision -> all-considerations map from
            ``expand.build_court_sibling_index``. When provided and enabled in
            CONFIG, court candidates are expanded to sibling considerations.

    No relevance-selection step is applied here yet — that is a separate stage.
    This stage is about maximising candidate-pool recall; the rich per-search
    logs let us measure recall@candidates downstream.

    Returns:
        Tuple of (citations, logs) where logs contains detailed execution info.
    """
    logs: list[dict] = []

    issues = decompose_query(query)
    if not issues:
        # Fallback: treat the whole query as a single broad issue across both tools.
        issues = [
            {"issue": "full query (decomposition failed)", "de_keywords": query, "type": "both"}
        ]
        logs.append({"type": "decompose", "status": "fallback_empty", "issues": issues})
    else:
        logs.append({"type": "decompose", "issues_count": len(issues), "issues": issues})

    if verbose:
        print(f"\nDecomposed into {len(issues)} issue(s):")
        for it in issues:
            print(f"  - [{it.get('type')}] {it.get('issue')} :: {it.get('de_keywords')}")

    # Candidate pool: citation -> details (best score kept). Order preserved separately.
    pool: dict[str, dict] = {}
    ordered_citations: list[str] = []

    for i, issue in enumerate(issues):
        keywords = (issue.get("de_keywords") or issue.get("issue") or "").strip()
        if not keywords:
            continue

        for tool_name in _tool_targets(issue.get("type")):
            tool = tools.get(tool_name)
            if tool is None:
                continue

            try:
                observation = tool(keywords)
            except Exception as exc:  # noqa: BLE001 — one failed search must not kill the query
                logs.append({
                    "type": "search_error",
                    "issue_index": i,
                    "tool": tool_name,
                    "query": keywords,
                    "error": str(exc),
                })
                continue

            results = tool.get_last_results() if hasattr(tool, "get_last_results") else []
            issue_citations: list[str] = []
            for doc in results:
                citation = doc.get("citation")
                if not citation:
                    continue
                issue_citations.append(citation)
                score = doc.get("_score")
                if citation not in pool:
                    pool[citation] = {
                        "citation": citation,
                        "score": score,
                        "text": (doc.get("text") or "")[:300],
                        "source": tool_name,
                        "issue_index": i,
                    }
                    ordered_citations.append(citation)
                elif score is not None and (
                    pool[citation]["score"] is None or score > pool[citation]["score"]
                ):
                    pool[citation]["score"] = score

            logs.append({
                "type": "search",
                "issue_index": i,
                "issue": issue.get("issue"),
                "type_target": issue.get("type"),
                "tool": tool_name,
                "query": keywords,
                "citations": issue_citations,
                "citations_count": len(issue_citations),
                "observation_trunc": observation[:500] if isinstance(observation, str) else "",
            })

            if verbose:
                print(
                    f"  [issue {i}] {tool_name} '{keywords[:50]}' -> "
                    f"{len(issue_citations)} citations"
                )

    # Sibling-consideration expansion: add every consideration of each retrieved
    # court decision (turns decision-level hits into exact-consideration hits).
    retrieved_count = len(ordered_citations)
    siblings_added = 0
    if sibling_index and CONFIG.get("enable_sibling_expansion", True):
        ordered_citations, siblings_added = expand_court_siblings(
            ordered_citations,
            sibling_index,
            max_per_decision=CONFIG.get("max_siblings_per_decision"),
        )
        logs.append({
            "type": "sibling_expansion",
            "retrieved_count": retrieved_count,
            "siblings_added": siblings_added,
            "expanded_pool_size": len(ordered_citations),
        })

    logs.append({
        "type": "summary",
        "issues_count": len(issues),
        "retrieved_count": retrieved_count,
        "siblings_added": siblings_added,
        "candidate_pool_size": len(ordered_citations),
        "citations": ordered_citations,
    })

    if verbose:
        print(
            f"\nCandidate pool: {len(ordered_citations)} unique citations "
            f"({retrieved_count} retrieved + {siblings_added} siblings)"
        )

    return ordered_citations, logs

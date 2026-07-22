import re
# from expand import expand_court_siblings
# from our_pipeline.llm.decompose import decompose_query
from omnilex.citations.normalizer import CitationNormalizer
from our_pipeline.llm.load_llm import llm
from our_pipeline.llm.prompts import AGENT_SYSTEM_PROMPT
from our_pipeline.constants import CONFIG

# Shared normalizer for parsing the model's Final Answer into canonical citations.
_normalizer = CitationNormalizer()


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


def parse_final_answer(text: str) -> list[str]:
    """Parse the model's Final Answer into citations.

    The agent is instructed to output a ``;``-separated list of citations. Each
    fragment is canonicalized with ``CitationNormalizer`` (which correctly strips
    ``lit.``/``Ziff.`` qualifiers and preserves decimal considerations like
    ``E. 6.2``). Fragments the normalizer cannot parse (e.g. ``SR`` citations or
    free-text prose) fall back to the regex extractor, keeping its ``SR`` support.

    Args:
        text: Final Answer text (the part after "Final Answer:" or the raw answer).

    Returns:
        Deduplicated list of citation strings.
    """
    if "Final Answer:" in text:
        text = text.split("Final Answer:")[-1]

    citations: list[str] = []
    seen: set[str] = set()

    def _add(citation: str) -> None:
        if citation and citation not in seen:
            seen.add(citation)
            citations.append(citation)

    fragments = [f.strip() for f in re.split(r"[;\n]", text) if f.strip()]
    for fragment in fragments:
        canonical = _normalizer.canonicalize(fragment)
        if canonical:
            _add(canonical)
        else:
            # SR citations / prose: defer to the regex extractor (keeps SR pattern).
            for c in extract_citations_from_text(fragment):
                _add(c)

    return citations


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


def _complete(conversation: str, max_tokens: int) -> str:
    """Call the agent LLM, retrying once with a larger budget if it returns empty.

    Agent calls run on the non-reasoning instruct model (``CONFIG["agent_model"]``);
    the default reasoning model can spend the whole token budget on hidden reasoning
    and return empty content, which would yield zero citations for the query.

    Raises:
        ValueError: propagated (e.g. context-window overflow) for the caller to handle.
    """
    model = CONFIG.get("agent_model")
    stop = ["Observation:", "[INST]", "</s>"]
    text = llm(
        conversation,
        max_tokens=max_tokens,
        temperature=CONFIG["temperature"],
        stop=stop,
        model=model,
    )["choices"][0]["text"]
    if not text.strip():
        text = llm(
            conversation,
            max_tokens=CONFIG.get("max_tokens_retry", max_tokens * 2),
            temperature=CONFIG["temperature"],
            stop=stop,
            model=model,
        )["choices"][0]["text"]
    return text


def run_agent(query: str, tools: dict[str, callable], verbose: bool = False) -> tuple[list[str], list[dict]]:
    """Run ReAct agent to retrieve citations.

    Predictions are the curated set the model lists in its ``Final Answer:`` (LLM
    final-selection), not the union of every retrieved hit. Retrieved citations are
    kept only as context/logs to avoid flooding predictions and destroying precision.

    Returns:
        Tuple of (citations, logs) where logs contains detailed execution info.
    """
    # Format with Mistral Instruct tags
    conversation = f"[INST] {AGENT_SYSTEM_PROMPT}\n\nQuery: {query}\n\nThought: [/INST]"
    final_citations: list[str] = []  # predictions (from Final Answer only)
    candidate_citations: list[str] = []  # retrieved hits, for logs / fallback only
    logs: list[dict] = []

    for iteration in range(CONFIG["max_iterations"]):
        # Truncate conversation if too long to avoid context window overflow
        max_conv_chars = CONFIG.get("max_conversation_chars", 28000)
        conversation = truncate_conversation(conversation, max_conv_chars)

        # Get LLM response with error handling for context overflow
        try:
            response = _complete(conversation, CONFIG["max_tokens"])
        except ValueError as e:
            error_str = str(e).lower()
            if "exceed context window" in error_str or "requested tokens" in error_str:
                # Aggressively truncate and retry once
                conversation = truncate_conversation(conversation, max_chars=20000)
                try:
                    response = _complete(conversation, CONFIG["max_tokens"])
                except ValueError as retry_error:
                    # Give up, return citations found so far
                    logs.append({
                        "type": "error",
                        "iteration": iteration + 1,
                        "error": f"Context overflow after retry: {retry_error}",
                    })
                    break
            else:
                raise

        # For subsequent turns, we need to handle the conversation format
        if iteration == 0:
            conversation = f"[INST] {AGENT_SYSTEM_PROMPT}\n\nQuery: {query} [/INST]\n\nThought:{response}"
        else:
            conversation += response

        # Log LLM output
        logs.append({
            "type": "sibling_expansion",
            "retrieved_count": retrieved_count,
            "siblings_added": siblings_added,
            "expanded_pool_size": len(ordered_citations),
        })

        if verbose:
            print(f"\n[Iteration {iteration + 1}] LLM output (trunc):")
            print(response[:500])

        # Parse all actions from response
        actions = parse_all_agent_actions(response)

        # Log parsed actions
        if actions:
            logs.append({
                "type": "parse",
                "iteration": iteration + 1,
                "actions_count": len(actions),
                "actions": actions,
            })
            if verbose:
                print(f"\n[Iteration {iteration + 1}] Parsed {len(actions)} action(s):")
                for action, action_input in actions:
                    print(f"  Action: {action}, Input: {action_input[:100]}")

        # Execute all actions
        observations = []
        for action, action_input in actions:
            action_lower = action.lower()

            if action_lower in tools:
                tool = tools[action_lower]
                try:
                    observation = tool(action_input)
                    # Retrieved citations are CONTEXT only, not predictions (precision).
                    obs_citations = tool.get_last_citations()
                except Exception as tool_exc:  # noqa: BLE001 - one bad search must not kill the query
                    observation = f"Tool error: {tool_exc}"
                    obs_citations = []
                    logs.append({
                        "type": "tool_error",
                        "iteration": iteration + 1,
                        "tool": action,
                        "error": str(tool_exc),
                    })
                candidate_citations.extend(obs_citations)

                # Truncate observation only for LLM conversation (preserve full data in logs)
                obs_truncated = truncate_observation_for_llm(observation, CONFIG["max_observation_chars"])
                observations.append(f"Tool {action_lower}: {obs_truncated}")

                # Log tool execution with full observation
                logs.append({
                    "type": "tool_execution",
                    "iteration": iteration + 1,
                    "tool": action,
                    "query": action_input,
                    "citations_found": obs_citations,
                    "citations_count": len(obs_citations),
                    "observation": observation,
                    "observation_trunc": observation[:500] if len(observation) > 500 else observation,
                })

                if verbose:
                    print(f"\n[Tool: {action}]")
                    print(f"  Query: {action_input}")
                    print(f"  Citations found: {len(obs_citations)}")
                    if obs_citations:
                        print(f"  Citations: {obs_citations[:5]}")
                    print(f"  Observation (trunc): {observation[:300]}")
            else:
                error_msg = f"Unknown tool '{action}'. Available: {', '.join(sorted(tools))}"
                observations.append(f"Tool {action_lower}: {error_msg}")
                logs.append({
                    "type": "tool_error",
                    "iteration": iteration + 1,
                    "tool": action,
                    "error": error_msg,
                })

        # Add all observations to conversation
        if observations:
            conversation += "\n" + "\n".join(observations) + "\n\n[INST] Continue your analysis. [/INST]\n\nThought:"

        # Check for final answer AFTER executing all actions
        if "Final Answer:" in response:
            final_text = response.split("Final Answer:")[-1].strip()
            final_citations.extend(parse_final_answer(final_text))

            logs.append({
                "type": "parse",
                "iteration": iteration + 1,
                "status": "final_answer_seen",
            })

            if verbose:
                print(f"\n[Iteration {iteration + 1}] Final Answer detected")
            break

        # If no actions and no final answer, the model stalled; stop and force a
        # final selection below instead of breaking with empty predictions.
        if not actions:
            logs.append({
                "type": "parse",
                "iteration": iteration + 1,
                "status": "no_actions_found",
            })
            break

    # If the loop ended without a Final Answer, force one grounded in the gathered
    # context so we still emit a curated (not flooded, not empty) prediction set.
    if not final_citations:
        conversation += (
            "\n\n[INST] Gib jetzt deine Final Answer aus: nur die relevanten Zitate "
            "aus den bisherigen Suchergebnissen, durch ';' getrennt. [/INST]\n\nFinal Answer:"
        )
        try:
            forced = _complete(conversation, CONFIG["max_tokens"])
        except ValueError:
            forced = ""
        final_citations.extend(parse_final_answer(forced))
        logs.append({
            "type": "parse",
            "status": "forced_final_selection",
            "citations_extracted": list(set(final_citations)),
        })

    # Deduplicate citations
    unique_citations = list(set(final_citations))
    n_candidates = len(set(candidate_citations))

    logs.append({
        "type": "summary",
        "total_iterations": len(logs),
        "candidate_citations": n_candidates,
        "total_citations": len(unique_citations),
        "citations": unique_citations,
    })

    if verbose:
        print("\n" + "="*50)
        print(f"Candidates retrieved: {n_candidates} | Selected: {len(unique_citations)}")
        for c in unique_citations:
            print(f"  - {c}")

    return unique_citations, logs

import re
from our_pipeline.llm.load_llm import llm
from our_pipeline.llm.prompts import AGENT_SYSTEM_PROMPT
from our_pipeline.constants import CONFIG


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
    input_pattern = r"Action Input:\s*(.+?)(?=\nAction:|$)"
    
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


def run_agent(query: str, tools: dict[str, callable], verbose: bool = False) -> tuple[list[str], list[dict]]:
    """Run ReAct agent to retrieve citations.
    
    Returns:
        Tuple of (citations, logs) where logs contains detailed execution information
    """
    # Format with Mistral Instruct tags
    conversation = f"[INST] {AGENT_SYSTEM_PROMPT}\n\nQuery: {query}\n\nThought: [/INST]"
    all_citations = []
    logs: list[dict] = []
    
    for iteration in range(CONFIG["max_iterations"]):
        # Truncate conversation if too long to avoid context window overflow
        max_conv_chars = CONFIG.get("max_conversation_chars", 28000)
        conversation = truncate_conversation(conversation, max_conv_chars)
        
        # Get LLM response with error handling for context overflow
        try:
            response = llm(
                conversation,
                max_tokens=CONFIG["max_tokens"],
                temperature=CONFIG["temperature"],
                stop=["Observation:", "[INST]", "</s>"],
            )["choices"][0]["text"]
        except ValueError as e:
            error_str = str(e).lower()
            if "exceed context window" in error_str or "requested tokens" in error_str:
                # Aggressively truncate and retry once
                conversation = truncate_conversation(conversation, max_chars=20000)
                try:
                    response = llm(
                        conversation,
                        max_tokens=CONFIG["max_tokens"],
                        temperature=CONFIG["temperature"],
                        stop=["Observation:", "[INST]", "</s>"],
                    )["choices"][0]["text"]
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
            "type": "llm_response",
            "iteration": iteration + 1,
            "response": response,
            "response_trunc": response[:500] if len(response) > 500 else response,
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
                observation = tool(action_input)
                
                # Extract citations from full observation (before truncation)
                obs_citations = tool.get_last_citations()
                all_citations.extend(obs_citations)
                
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
                error_msg = f"Unknown tool '{action}'. Available: search_laws, search_courts"
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
            citations = extract_citations_from_text(final_text)
            all_citations.extend(citations)
            
            logs.append({
                "type": "parse",
                "iteration": iteration + 1,
                "status": "final_answer_seen",
            })
            
            if verbose:
                print(f"\n[Iteration {iteration + 1}] Final Answer detected")
            break
        
        # If no actions found and no final answer, try to extract citations from response
        if not actions and "Final Answer:" not in response:
            citations = extract_citations_from_text(response)
            all_citations.extend(citations)
            logs.append({
                "type": "parse",
                "iteration": iteration + 1,
                "status": "no_actions_found",
                "citations_extracted": citations,
            })
            break
    
    # Deduplicate citations
    unique_citations = list(set(all_citations))
    
    logs.append({
        "type": "summary",
        "total_iterations": len(logs),
        "total_citations": len(unique_citations),
        "citations": unique_citations,
    })
    
    if verbose:
        print("\n" + "="*50)
        print("Found citations:")
        for c in unique_citations:
            print(f"  - {c}")
    
    return unique_citations, logs
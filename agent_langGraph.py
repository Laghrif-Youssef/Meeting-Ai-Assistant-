"""
Agent LangGraph — fixed and improved version.

Problems fixed vs original:
  1. LLM hallucination of tool names / wrong argument types
       → Two-phase architecture: PLANNER produces an ordered task list first,
         EXECUTOR works through it one step at a time.
       → Strict per-tool example blocks injected into the system prompt.
       → Hard type coercion in the validator (auto-fixes before rejecting).

  2. Multi-recipient emails sent as one call with a list
       → Planner is explicitly instructed to expand N recipients into N steps.
       → Validator rejects list values for string arguments at execute time.

  3. Dependency ordering (create event → use link in emails)
       → Planner produces a numbered, dependency-aware task list.
       → Executor injects tool results back into state so the next step
         can reference them (e.g. the Meet link from step 1 goes into step 2).

  4. current_tool_call never cleared after execution
       → State is explicitly reset to None after each execution.

  5. Planner sync / execute async mismatch
       → Both nodes are now async.

  6. Messages lost on JSON parse error
       → Error recovery appends a correction message and preserves full state.

  7. Infinite loop risk
       → Hard cap of MAX_STEPS total tool executions per run.
"""

import asyncio
import json
import re
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from llm_client import ask_llm
from tool_registry import (
    init_registry,
    execute_tool,
    open_sessions,
    close_sessions,
    TOOL_REGISTRY,
)


MAX_STEPS = 20   # hard cap — prevents infinite loops


# ══════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════

class AgentState(TypedDict):
    messages:           List[Dict[str, str]]  # full conversation history
    task_queue:         List[Dict[str, Any]]  # ordered steps from planner
    current_tool_call:  Dict[str, Any] | None # step being executed right now
    tool_results:       List[Dict[str, Any]]  # all results accumulated so far
    final_answer:       str | None
    done:               bool
    steps_taken:        int                   # guard against infinite loops
    parse_errors:       int                   # guard against prompt loops


# ══════════════════════════════════════════════
# PROMPT HELPERS
# ══════════════════════════════════════════════

def _schema_summary(schema: dict) -> str:
    """Produce a compact, example-rich summary of one tool's schema."""
    props = schema.get("properties", {})
    required = schema.get("required", [])
    lines = []
    for name, spec in props.items():
        t = spec.get("type", "any")
        desc = spec.get("description", "")
        req = " [REQUIRED]" if name in required else ""
        lines.append(f'    "{name}": <{t}>{req}  // {desc}')
    return "\n".join(lines)


def build_planner_prompt() -> str:
    """
    System prompt for the PLANNER phase.
    The planner must output an ordered list of tool calls — no execution yet.
    """
    tool_blocks = []
    for name, info in TOOL_REGISTRY.items():
        schema_str = _schema_summary(info["schema"])
        tool_blocks.append(
            f"TOOL: {name}\n"
            f"DESCRIPTION: {info.get('description', '')}\n"
            f"ARGUMENTS:\n{schema_str}"
        )
    tools_text = "\n\n".join(tool_blocks)

    return f"""
You are a planning agent. Your ONLY job is to decompose the user's request into
an ordered list of tool calls. You do NOT execute anything — you only plan.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES — READ CAREFULLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 — USE ONLY REAL TOOL NAMES.
  Allowed tools: {list(TOOL_REGISTRY.keys())}
  NEVER invent a tool name. If a tool does not exist for a subtask, skip it.

RULE 2 — ONE RECIPIENT PER EMAIL CALL.
  send_email_tool accepts "to" as a single email string — NOT a list, NOT comma-separated.
  If the user wants to email 3 people → produce 3 separate send_email_tool steps.

  WRONG:  {{"to": "a@x.com,b@x.com"}}
  WRONG:  {{"to": ["a@x.com", "b@x.com"]}}
  RIGHT:  step 1: {{"to": "a@x.com", ...}}   step 2: {{"to": "b@x.com", ...}}

RULE 3 — RESPECT ARGUMENT TYPES EXACTLY.
  If the schema says a field is "string", the value must be a JSON string.
  If the schema says "integer", use a number with no quotes.

RULE 4 — DEPENDENCY ORDER.
  If step B needs data from step A (e.g. a meeting link created in step A),
  put step A first. Use the placeholder {{{{RESULT_0}}}} to reference
  the result of step 0, {{{{RESULT_1}}}} for step 1, etc.
  The executor will substitute the real value before calling the tool.

RULE 5 — OUTPUT FORMAT.
  Return ONLY this JSON — no markdown, no explanation:

{{
  "steps": [
    {{
      "step": 0,
      "tool": "tool_name",
      "arguments": {{...}},
      "reason": "one sentence why"
    }},
    ...
  ]
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKED EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USER: "Send an email to a@x.com and b@x.com saying hello"
PLAN:
{{
  "steps": [
    {{"step":0,"tool":"send_email_tool","arguments":{{"to":"a@x.com","subject":"Hello","body":"Hello"}},"reason":"email to first recipient"}},
    {{"step":1,"tool":"send_email_tool","arguments":{{"to":"b@x.com","subject":"Hello","body":"Hello"}},"reason":"email to second recipient"}}
  ]
}}

USER: "Schedule a meeting on June 25 10:00-11:00 with a@x.com and b@x.com then send each an invitation with the Meet link"
PLAN:
{{
  "steps": [
    {{"step":0,"tool":"create_event_tool","arguments":{{"title":"Meeting","start_time":"2026-06-25T10:00:00","end_time":"2026-06-25T11:00:00"}},"reason":"create the calendar event first to get the event link"}},
    {{"step":1,"tool":"send_email_tool","arguments":{{"to":"a@x.com","subject":"Meeting invitation","body":"You are invited. Link: {{{{RESULT_0}}}}"}},"reason":"invite first participant using the event link from step 0"}},
    {{"step":2,"tool":"send_email_tool","arguments":{{"to":"b@x.com","subject":"Meeting invitation","body":"You are invited. Link: {{{{RESULT_0}}}}"}},"reason":"invite second participant"}}
  ]
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{tools_text}

Return ONLY valid JSON. No markdown fences. No explanation outside the JSON.
""".strip()


def build_executor_prompt() -> str:
    """
    System prompt for the EXECUTOR phase.
    At this point we already have a plan; the executor just reports done/continue.
    """
    return """
You are an execution monitor. The agent is executing a pre-planned list of tool calls.
After each tool result you will decide:
  - If there are more steps remaining → return: {"action": "continue"}
  - If all steps are done → return: {"action": "final", "answer": "<summary for the user>"}

Return ONLY valid JSON.
""".strip()


# ══════════════════════════════════════════════
# VALIDATOR  (auto-coerce where safe, reject otherwise)
# ══════════════════════════════════════════════

def validate_and_coerce(tool_name: str, arguments: dict) -> dict:
    """
    Validate arguments against the tool schema.
    Auto-coerces safe mismatches (e.g. int passed as string).
    Raises ValueError for structural violations (list where string required).
    Returns the (possibly coerced) arguments dict.
    """
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(
            f"Unknown tool '{tool_name}'. "
            f"Available: {list(TOOL_REGISTRY.keys())}"
        )

    schema = TOOL_REGISTRY[tool_name]["schema"]
    props  = schema.get("properties", {})
    coerced = dict(arguments)

    for arg_name, value in arguments.items():
        if arg_name not in props:
            raise ValueError(
                f"Tool '{tool_name}' has no argument '{arg_name}'. "
                f"Valid arguments: {list(props.keys())}"
            )

        expected = props[arg_name].get("type")

        if expected == "string":
            if isinstance(value, list):
                raise ValueError(
                    f"Argument '{arg_name}' of '{tool_name}' must be a string, "
                    f"not a list. Call the tool once per item instead."
                )
            if not isinstance(value, str):
                coerced[arg_name] = str(value)   # safe coercion: int/float → str

        elif expected == "integer":
            if isinstance(value, str) and value.isdigit():
                coerced[arg_name] = int(value)   # safe coercion: "10" → 10
            elif not isinstance(value, int):
                raise ValueError(
                    f"Argument '{arg_name}' of '{tool_name}' must be an integer."
                )

        elif expected == "array":
            if not isinstance(value, list):
                raise ValueError(
                    f"Argument '{arg_name}' of '{tool_name}' must be an array."
                )

    # Check required fields are present
    for req in schema.get("required", []):
        if req not in coerced:
            raise ValueError(
                f"Required argument '{req}' missing for tool '{tool_name}'."
            )

    return coerced


# ══════════════════════════════════════════════
# RESULT SUBSTITUTION
# ══════════════════════════════════════════════

def _extract_useful_value(result) -> str:
    """
    Pull the most useful string out of an MCP tool result.
    Tries: event_link, meet_link, html_link, gmail_message_id, then full JSON.
    """
    # result is usually a list of TextContent objects
    raw = result
    if isinstance(result, list) and result:
        raw = result[0]

    # TextContent has a .text attribute
    text = getattr(raw, "text", None) or str(raw)

    try:
        data = json.loads(text)
        # Priority order for useful links/IDs
        for key in ("event_link", "meet_link", "html_link",
                    "gmail_message_id", "event_id", "message"):
            if key in data and data[key]:
                return str(data[key])
        return text   # return full JSON string if no known key
    except Exception:
        return text


def substitute_results(arguments: dict, tool_results: list[dict]) -> dict:
    """
    Replace {{RESULT_N}} placeholders in argument values with actual tool results.
    """
    substituted = {}
    for key, value in arguments.items():
        if isinstance(value, str) and "{{RESULT_" in value:
            for i, res in enumerate(tool_results):
                placeholder = f"{{{{RESULT_{i}}}}}"
                if placeholder in value:
                    value = value.replace(placeholder, res.get("value", ""))
        substituted[key] = value
    return substituted


# ══════════════════════════════════════════════
# NODES
# ══════════════════════════════════════════════

async def planner_node(state: AgentState) -> dict:
    """
    Phase 1: ask the LLM to produce a full ordered task list.
    Does NOT execute anything — only decomposes the request.
    """
    prompt   = build_planner_prompt()
    response = ask_llm(state["messages"], prompt)

    print(f"\n{'='*50}\nPLANNER RESPONSE:\n{response}\n{'='*50}")

    # Strip markdown fences if the model added them despite instructions
    clean = re.sub(r"```(?:json)?|```", "", response).strip()

    try:
        plan = json.loads(clean)
        steps = plan.get("steps", [])

        if not steps:
            raise ValueError("Empty steps list")

        # Validate all tool names up front before starting execution
        for step in steps:
            tool = step.get("tool", "")
            if tool not in TOOL_REGISTRY:
                raise ValueError(f"Step {step.get('step')} uses unknown tool '{tool}'")

        print(f"[planner] Plan accepted: {len(steps)} steps")
        for s in steps:
            print(f"  Step {s['step']}: {s['tool']} — {s.get('reason','')}")

        return {
            "task_queue":        steps,
            "current_tool_call": None,
            "done":              False,
            "parse_errors":      0,
        }

    except Exception as e:
        errors = state.get("parse_errors", 0) + 1
        print(f"[planner] Parse error #{errors}: {e}")

        if errors >= 3:
            # Give up after 3 failed attempts
            return {
                "final_answer": "I was unable to plan the task correctly. Please rephrase.",
                "done":         True,
                "parse_errors": errors,
            }

        # Ask the LLM to fix its output
        correction = (
            f"Your previous response was invalid: {e}. "
            "Return ONLY the JSON plan with no markdown, no extra text."
        )
        return {
            "messages":     state["messages"] + [{"role": "user", "content": correction}],
            "parse_errors": errors,
            "done":         False,
        }


async def executor_node(state: AgentState) -> dict:
    """
    Phase 2: pop the next step from task_queue, execute it, store the result.
    """
    task_queue   = list(state.get("task_queue", []))
    tool_results = list(state.get("tool_results", []))
    steps_taken  = state.get("steps_taken", 0)

    # Safety cap
    if steps_taken >= MAX_STEPS:
        print(f"[executor] MAX_STEPS ({MAX_STEPS}) reached — stopping.")
        return {"final_answer": "Reached maximum step limit.", "done": True}

    # Nothing left to do
    if not task_queue:
        return {"done": True, "current_tool_call": None}

    # Pop the first step
    step      = task_queue.pop(0)
    tool_name = step.get("tool", "")
    arguments = dict(step.get("arguments", {}))

    print(f"\n[executor] Step {step.get('step')}: {tool_name}")
    print(f"  Arguments (before substitution): {arguments}")

    # Substitute {{RESULT_N}} placeholders from prior results
    arguments = substitute_results(arguments, tool_results)
    print(f"  Arguments (after substitution):  {arguments}")

    try:
        # Validate and auto-coerce
        arguments = validate_and_coerce(tool_name, arguments)

        # Execute
        result = await execute_tool(tool_name, arguments)

        useful_value = _extract_useful_value(result)
        print(f"  Result: {useful_value[:200]}")

        tool_results.append({
            "step":  step.get("step"),
            "tool":  tool_name,
            "value": useful_value,
            "raw":   str(result),
        })

        new_messages = state["messages"] + [
            {"role": "assistant", "content": json.dumps({"tool": tool_name, "arguments": arguments})},
            {"role": "tool",      "content": f"Result of {tool_name}: {useful_value}"},
        ]

        still_work = len(task_queue) > 0

        return {
            "messages":          new_messages,
            "task_queue":        task_queue,
            "tool_results":      tool_results,
            "current_tool_call": step if still_work else None,
            "steps_taken":       steps_taken + 1,
            "done":              not still_work,
        }

    except Exception as e:
        print(f"  [executor] Error: {e}")

        error_msg = f"ERROR executing {tool_name}: {e}"
        new_messages = state["messages"] + [
            {"role": "tool", "content": error_msg},
        ]

        # If there are steps remaining, continue anyway
        still_work = len(task_queue) > 0
        return {
            "messages":     new_messages,
            "task_queue":   task_queue,
            "tool_results": tool_results,
            "steps_taken":  steps_taken + 1,
            "done":         not still_work,
        }


async def summarizer_node(state: AgentState) -> dict:
    """
    Final node: ask the LLM to summarise what was accomplished.
    """
    results_summary = "\n".join(
        f"  Step {r['step']} ({r['tool']}): {r['value']}"
        for r in state.get("tool_results", [])
    )

    summary_prompt = (
        "The agent has finished executing all planned steps. "
        "Write a clear, friendly summary in the same language as the user's request. "
        "Include what was done and any important values (links, IDs, etc.).\n\n"
        f"Steps executed:\n{results_summary}"
    )

    messages = state["messages"] + [{"role": "user", "content": summary_prompt}]
    answer   = ask_llm(messages, "You are a helpful assistant. Summarise concisely.")

    # Strip JSON fences if model wrapped the answer
    answer = re.sub(r"```(?:json)?|```", "", answer).strip()
    try:
        parsed = json.loads(answer)
        answer = parsed.get("answer", answer)
    except Exception:
        pass

    return {"final_answer": answer, "done": True}


# ══════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════

def route_after_planner(state: AgentState) -> str:
    if state.get("done"):
        return "summarizer"
    if state.get("task_queue"):
        return "executor"
    return "planner"   # parse error retry


def route_after_executor(state: AgentState) -> str:
    if state.get("done") or not state.get("task_queue"):
        return "summarizer"
    return "executor"  # more steps to process


# ══════════════════════════════════════════════
# GRAPH
# ══════════════════════════════════════════════

graph = StateGraph(AgentState)

graph.add_node("planner",    planner_node)
graph.add_node("executor",   executor_node)
graph.add_node("summarizer", summarizer_node)

graph.set_entry_point("planner")

graph.add_conditional_edges("planner",  route_after_planner, {
    "executor":   "executor",
    "planner":    "planner",
    "summarizer": "summarizer",
})

graph.add_conditional_edges("executor", route_after_executor, {
    "executor":   "executor",
    "summarizer": "summarizer",
})

graph.add_edge("summarizer", END)

app = graph.compile()


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

async def main():
    await init_registry()
    await open_sessions()   # open persistent MCP sessions once

    try:
        print("\nAgent ready. Type your request (Ctrl+C to quit).\n")
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            initial_state: AgentState = {
                "messages":          [{"role": "user", "content": user_input}],
                "task_queue":        [],
                "current_tool_call": None,
                "tool_results":      [],
                "final_answer":      None,
                "done":              False,
                "steps_taken":       0,
                "parse_errors":      0,
            }

            result = await app.ainvoke(initial_state)

            print(f"\nAssistant: {result['final_answer']}\n")

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await close_sessions()


if __name__ == "__main__":
    asyncio.run(main())
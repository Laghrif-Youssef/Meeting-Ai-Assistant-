"""
Agent LangGraph — v3

Key changes vs v2:
  - Clarification logic moved entirely to Python (pre_flight_check).
    The LLM planner no longer needs to detect missing fields — it just plans.
  - Planner prompt is much shorter → 7B models follow it reliably.
  - Context dict from pre_flight_check injected directly into planner prompt
    so metadata values are hardcoded and the model cannot invent them.
"""

import asyncio
import json
import re
from datetime import datetime
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


MAX_STEPS = 20


# ══════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════

class AgentState(TypedDict):
    messages:                List[Dict[str, str]]
    task_queue:              List[Dict[str, Any]]
    current_tool_call:       Dict[str, Any] | None
    tool_results:            List[Dict[str, Any]]
    final_answer:            str | None
    done:                    bool
    steps_taken:             int
    parse_errors:            int
    needs_clarification:     bool
    clarification_question:  str | None


# ══════════════════════════════════════════════
# PRE-FLIGHT CHECK  (pure Python — no LLM)
# ══════════════════════════════════════════════

# Tools that need meeting metadata to produce a useful result
PV_TOOLS = {"generate_pv_pdf_tool", "generate_pv_tool"}

def pre_flight_check(user_message: str) -> tuple[bool, str, dict]:
    """
    Inspect the user's message BEFORE calling the LLM.
    Returns (needs_clarification, question, context_dict).

    If the message mentions a PV/transcription tool but is missing metadata,
    returns needs_clarification=True and the question to ask.
    Otherwise returns False with whatever context could be extracted.
    """
    msg_lower = user_message.lower()

    # Does this request involve a PV/transcription tool?
    pv_requested = any(kw in msg_lower for kw in (
        "pv", "procès-verbal", "proces-verbal", "transcri", "audio",
        "meeting", "réunion", "reunion", "generate_pv", "wav", "mp3", "m4a"
    ))

    if not pv_requested:
        return False, "", {}

    # Try to extract whatever is already present
    context = {}
    missing = []

    # meeting_title — look for quoted strings or explicit labels
    title_match = re.search(
        r'(?:title|titre|réunion|reunion)[:\s"\']+([^",\n]+)', user_message, re.IGNORECASE
    )
    if title_match:
        context["meeting_title"] = title_match.group(1).strip()
    else:
        missing.append("le titre de la réunion")

    # meeting_date — look for common date patterns
    date_match = re.search(
        r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{1,2}\s+\w+\s+\d{4})\b',
        user_message
    )
    if date_match:
        context["meeting_date"] = date_match.group(1)
    else:
        missing.append("la date")

    # meeting_time — look for time patterns
    time_match = re.search(r'\b(\d{1,2}[h:]\d{0,2})\b', user_message, re.IGNORECASE)
    if time_match:
        context["meeting_time"] = time_match.group(1).replace("h", ":").rstrip(":")
    else:
        missing.append("l'heure")

    # location
    loc_match = re.search(
        r'(?:lieu|location|salle|meet|teams|zoom|plateforme)[:\s"\']+([^",\n]+)',
        user_message, re.IGNORECASE
    )
    if loc_match:
        context["location"] = loc_match.group(1).strip()
    elif "google meet" in msg_lower:
        context["location"] = "Google Meet"
    elif "teams" in msg_lower:
        context["location"] = "Microsoft Teams"
    elif "zoom" in msg_lower:
        context["location"] = "Zoom"
    else:
        missing.append("le lieu ou la plateforme (ex: Google Meet)")

    if missing:
        question = (
            "Pour générer le PV, j'ai besoin de quelques informations :\n"
            + "\n".join(f"  • {m}" for m in missing)
        )
        return True, question, context

    return False, "", context


def parse_user_reply(reply: str, existing_context: dict) -> dict:
    """
    After the user answers the clarification question,
    extract missing fields from their reply and merge with existing context.
    """
    context = dict(existing_context)

    # title — anything that looks like a meeting name
    if "meeting_title" not in context:
        # First quoted string, or first sentence before a comma
        title_match = re.search(r'"([^"]+)"', reply) or re.search(r'^([^,\n]+)', reply)
        if title_match:
            context["meeting_title"] = title_match.group(1).strip()

    # date
    if "meeting_date" not in context:
        date_match = re.search(
            r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{1,2}\s+\w+\s+\d{4})\b', reply
        )
        if date_match:
            context["meeting_date"] = date_match.group(1)
        else:
            context["meeting_date"] = datetime.now().strftime("%d/%m/%Y")

    # time
    if "meeting_time" not in context:
        time_match = re.search(r'\b(\d{1,2}[h:]\d{0,2})\b', reply, re.IGNORECASE)
        if time_match:
            context["meeting_time"] = time_match.group(1).replace("h", ":").rstrip(":")

    # location
    if "location" not in context:
        reply_lower = reply.lower()
        if "google meet" in reply_lower:
            context["location"] = "Google Meet"
        elif "teams" in reply_lower:
            context["location"] = "Microsoft Teams"
        elif "zoom" in reply_lower:
            context["location"] = "Zoom"
        else:
            loc_match = re.search(r'(?:salle|lieu|room|à)\s+([^\.,\n]+)', reply, re.IGNORECASE)
            if loc_match:
                context["location"] = loc_match.group(1).strip()
            else:
                # Use the whole reply as location if nothing else matched
                context["location"] = reply.strip()

    return context


# ══════════════════════════════════════════════
# PROMPT HELPERS
# ══════════════════════════════════════════════

def _schema_summary(schema: dict) -> str:
    props    = schema.get("properties", {})
    required = schema.get("required", [])
    lines    = []
    for name, spec in props.items():
        t    = spec.get("type", "any")
        desc = spec.get("description", "")
        req  = " [REQUIRED]" if name in required else ""
        lines.append(f'    "{name}": <{t}>{req}  // {desc}')
    return "\n".join(lines)


def build_planner_prompt(context: dict | None = None) -> str:
    """
    Short, focused planner prompt.
    context: pre-filled metadata values injected as hard constraints.
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

    # Hard-inject collected metadata so the model cannot invent values
    context_block = ""
    if context:
        # Extract the audio file path from the messages if present — pass it directly
        context_block = (
            "\nCONTEXT — MANDATORY, DO NOT DEVIATE:\n"
            + "\n".join(f'  {k} = "{v}"' for k, v in context.items())
            + "\n\n"
            "MANDATORY RULE FOR PV GENERATION:\n"
            "  When the user wants a PV PDF from an audio file, use ONLY generate_pv_pdf_tool\n"
            "  in a SINGLE step. Do NOT call transcribe_tool or generate_pv_tool separately.\n"
            "  generate_pv_pdf_tool already handles transcription + extraction + PDF internally.\n"
            "  The audio_file argument must be the exact file path from the user's message.\n"
            "  All other arguments come from the CONTEXT values above.\n"
        )

    return f"""
You are a planning agent. Decompose the user request into an ordered list of tool calls.

RULES:
1. Only use these tool names: {list(TOOL_REGISTRY.keys())}
2. send_email_tool "to" must be ONE email string. N recipients = N separate steps.
3. If step B needs a value from step A, write {{{{RESULT_0}}}} in step B's argument.
4. Return ONLY valid JSON — no markdown, no explanation outside JSON.
{context_block}
OUTPUT:
{{
  "steps": [
    {{"step": 0, "tool": "tool_name", "arguments": {{...}}, "reason": "why"}}
  ]
}}

EXAMPLES:

User: "Send hello to a@x.com and b@x.com"
{{"steps":[
  {{"step":0,"tool":"send_email_tool","arguments":{{"to":"a@x.com","subject":"Hello","body":"Hello"}},"reason":"first"}},
  {{"step":1,"tool":"send_email_tool","arguments":{{"to":"b@x.com","subject":"Hello","body":"Hello"}},"reason":"second"}}
]}}

User: "Schedule meeting June 25 10-11 with a@x.com and b@x.com, send invites"
{{"steps":[
  {{"step":0,"tool":"create_event_tool","arguments":{{"title":"Meeting","start_time":"2026-06-25T10:00:00","end_time":"2026-06-25T11:00:00"}},"reason":"create event"}},
  {{"step":1,"tool":"send_email_tool","arguments":{{"to":"a@x.com","subject":"Invitation","body":"Join: {{{{RESULT_0}}}}"}},"reason":"invite a"}},
  {{"step":2,"tool":"send_email_tool","arguments":{{"to":"b@x.com","subject":"Invitation","body":"Join: {{{{RESULT_0}}}}"}},"reason":"invite b"}}
]}}

User: "generate pv pdf from audio C:/meeting.wav"  (with context: meeting_title="Réunion Q2", meeting_date="25/06/2026", meeting_time="10:00", location="Google Meet")
{{"steps":[
  {{"step":0,"tool":"generate_pv_pdf_tool","arguments":{{"audio_file":"C:/meeting.wav","meeting_title":"Réunion Q2","meeting_date":"25/06/2026","meeting_time":"10:00","location":"Google Meet"}},"reason":"single tool handles transcription + extraction + PDF"}}
]}}

AVAILABLE TOOLS:
{tools_text}
""".strip()


# ══════════════════════════════════════════════
# VALIDATOR
# ══════════════════════════════════════════════

def validate_and_coerce(tool_name: str, arguments: dict) -> dict:
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(
            f"Unknown tool '{tool_name}'. Available: {list(TOOL_REGISTRY.keys())}"
        )

    schema  = TOOL_REGISTRY[tool_name]["schema"]
    props   = schema.get("properties", {})
    coerced = dict(arguments)

    for arg_name, value in arguments.items():
        if arg_name not in props:
            raise ValueError(
                f"Tool '{tool_name}' has no argument '{arg_name}'. "
                f"Valid: {list(props.keys())}"
            )
        expected = props[arg_name].get("type")

        if expected == "string":
            if isinstance(value, list):
                raise ValueError(
                    f"'{arg_name}' of '{tool_name}' must be a string, not a list. "
                    "Call the tool once per item."
                )
            if not isinstance(value, str):
                coerced[arg_name] = str(value)

        elif expected == "integer":
            if isinstance(value, str) and value.isdigit():
                coerced[arg_name] = int(value)
            elif not isinstance(value, int):
                raise ValueError(f"'{arg_name}' of '{tool_name}' must be an integer.")

        elif expected == "array":
            if not isinstance(value, list):
                raise ValueError(f"'{arg_name}' of '{tool_name}' must be an array.")

    for req in schema.get("required", []):
        if req not in coerced:
            raise ValueError(f"Required argument '{req}' missing for '{tool_name}'.")

    return coerced


# ══════════════════════════════════════════════
# RESULT EXTRACTION & SUBSTITUTION
# ══════════════════════════════════════════════

def _extract_useful_value(result) -> str:
    raw  = result[0] if isinstance(result, list) and result else result
    text = getattr(raw, "text", None) or str(raw)
    try:
        data = json.loads(text)
        # For PV tool results, return the full JSON so the summarizer
        # can report the pdf_path, participants, actions etc.
        if "pdf_path" in data:
            return text   # return full JSON — summarizer will format it nicely
        # For other tools, extract the single most useful value
        for key in ("event_link", "meet_link", "html_link",
                    "gmail_message_id", "event_id", "message", "transcript"):
            if key in data and data[key]:
                return str(data[key])
        return text
    except Exception:
        return text


def substitute_results(arguments: dict, tool_results: list) -> dict:
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
    context = state.get("context", {})
    prompt  = build_planner_prompt(context if context else None)
    response = ask_llm(state["messages"], prompt)

    print(f"\n{'='*50}\nPLANNER RESPONSE:\n{response}\n{'='*50}")

    clean = re.sub(r"```(?:json)?|```", "", response).strip()

    try:
        plan  = json.loads(clean)
        steps = plan.get("steps", [])

        if not steps:
            raise ValueError("Empty steps list")

        for step in steps:
            tool = step.get("tool", "")
            if tool not in TOOL_REGISTRY:
                raise ValueError(f"Unknown tool '{tool}' in step {step.get('step')}")

        print(f"[planner] Plan: {len(steps)} steps")
        for s in steps:
            print(f"  Step {s['step']}: {s['tool']} — {s.get('reason','')}")

        return {
            "task_queue":            steps,
            "current_tool_call":     None,
            "needs_clarification":   False,
            "clarification_question":None,
            "done":                  False,
            "parse_errors":          0,
        }

    except Exception as e:
        errors = state.get("parse_errors", 0) + 1
        print(f"[planner] Parse error #{errors}: {e}")

        if errors >= 3:
            return {
                "final_answer": "Unable to plan the task. Please rephrase.",
                "done":         True,
                "parse_errors": errors,
            }

        return {
            "messages":     state["messages"] + [{
                "role": "user",
                "content": f"Invalid JSON: {e}. Return ONLY the JSON plan."
            }],
            "parse_errors": errors,
            "done":         False,
        }


async def executor_node(state: AgentState) -> dict:
    task_queue   = list(state.get("task_queue", []))
    tool_results = list(state.get("tool_results", []))
    steps_taken  = state.get("steps_taken", 0)

    if steps_taken >= MAX_STEPS:
        return {"final_answer": "Reached maximum step limit.", "done": True}

    if not task_queue:
        return {"done": True, "current_tool_call": None}

    step      = task_queue.pop(0)
    tool_name = step.get("tool", "")
    arguments = dict(step.get("arguments", {}))

    print(f"\n[executor] Step {step.get('step')}: {tool_name}")
    print(f"  Args (before): {arguments}")

    arguments = substitute_results(arguments, tool_results)
    print(f"  Args (after):  {arguments}")

    try:
        arguments = validate_and_coerce(tool_name, arguments)
        result    = await execute_tool(tool_name, arguments)

        useful_value = _extract_useful_value(result)
        print(f"  Result: {useful_value[:300]}")

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
        still_work = len(task_queue) > 0
        return {
            "messages":     state["messages"] + [{"role": "tool", "content": f"ERROR: {e}"}],
            "task_queue":   task_queue,
            "tool_results": tool_results,
            "steps_taken":  steps_taken + 1,
            "done":         not still_work,
        }


async def summarizer_node(state: AgentState) -> dict:
    results_summary = "\n".join(
        f"  Step {r['step']} ({r['tool']}): {r['value']}"
        for r in state.get("tool_results", [])
    )

    summary_prompt = (
        "The agent finished all steps. "
        "Write a clear, friendly summary in the same language as the user's request. "
        "Include what was done and any important values (links, file paths, IDs).\n\n"
        f"Steps executed:\n{results_summary}"
    )

    answer = ask_llm(
        state["messages"] + [{"role": "user", "content": summary_prompt}],
        "You are a helpful assistant. Summarise concisely."
    )
    answer = re.sub(r"```(?:json)?|```", "", answer).strip()
    try:
        parsed = json.loads(answer)
        answer = parsed.get("answer", parsed.get("summary", str(parsed)))
    except Exception:
        pass

    return {"final_answer": answer, "done": True}


async def clarifier_node(state: AgentState) -> dict:
    return {
        "needs_clarification":    True,
        "clarification_question": state.get("clarification_question"),
        "done":                   False,
    }


# ══════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════

def route_after_planner(state: AgentState) -> str:
    if state.get("done"):              return "summarizer"
    if state.get("needs_clarification"): return "clarifier"
    if state.get("task_queue"):        return "executor"
    return "planner"


def route_after_executor(state: AgentState) -> str:
    if state.get("done") or not state.get("task_queue"):
        return "summarizer"
    return "executor"


# ══════════════════════════════════════════════
# GRAPH
# ══════════════════════════════════════════════

graph = StateGraph(AgentState)
graph.add_node("planner",    planner_node)
graph.add_node("clarifier",  clarifier_node)
graph.add_node("executor",   executor_node)
graph.add_node("summarizer", summarizer_node)
graph.set_entry_point("planner")

graph.add_conditional_edges("planner", route_after_planner, {
    "clarifier":  "clarifier",
    "executor":   "executor",
    "planner":    "planner",
    "summarizer": "summarizer",
})
graph.add_conditional_edges("executor", route_after_executor, {
    "executor":   "executor",
    "summarizer": "summarizer",
})
graph.add_edge("clarifier",  END)
graph.add_edge("summarizer", END)

app = graph.compile()


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

async def main():
    await init_registry()
    await open_sessions()

    try:
        print("\nAgent ready. Type your request (Ctrl+C to quit).\n")
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            messages: List[Dict[str, str]] = [{"role": "user", "content": user_input}]
            context: dict = {}

            while True:
                # ── Pre-flight check (Python, no LLM) ────
                needs_clarif, question, partial_context = pre_flight_check(
                    # Check against the full latest user message
                    messages[-1]["content"]
                )
                context = {**context, **partial_context}  # accumulate across rounds

                if needs_clarif:
                    print(f"\nAssistant: {question}")
                    user_reply = input("You: ").strip()
                    if not user_reply:
                        continue

                    # Merge newly provided values into context
                    context = parse_user_reply(user_reply, context)

                    messages = messages + [
                        {"role": "assistant", "content": question},
                        {"role": "user",      "content": user_reply},
                    ]
                    # Loop back — check again if anything is still missing
                    continue

                # ── All info available → run the agent ───
                state: AgentState = {
                    "messages":               messages,
                    "task_queue":             [],
                    "current_tool_call":      None,
                    "tool_results":           [],
                    "final_answer":           None,
                    "done":                   False,
                    "steps_taken":            0,
                    "parse_errors":           0,
                    "needs_clarification":    False,
                    "clarification_question": None,
                    "context":                context,
                }

                result = await app.ainvoke(state)
                print(f"\nAssistant: {result.get('final_answer', '(no answer)')}\n")
                break

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await close_sessions()


if __name__ == "__main__":
    asyncio.run(main())
"""part_d/agent.py - ReAct agent execution, safety limits, and resumable state."""
import os
import json
import time
import mock_llm
from part_d.tools import TOOLS_SPEC, TOOL_MAP

STATE_DIR = "state"
os.makedirs(STATE_DIR, exist_ok=True)

def save_checkpoint(run_id, scenario_id, messages, tokens_used, step_number, tools_executed, steps, state_dir=STATE_DIR):
    os.makedirs(state_dir, exist_ok=True)
    cp_path = os.path.join(state_dir, f"{run_id}.json")
    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": step_number,
        "tools_executed": tools_executed,
        "steps": steps
    }
    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)

def load_checkpoint(run_id, state_dir=STATE_DIR):
    cp_path = os.path.join(state_dir, f"{run_id}.json")
    if not os.path.exists(cp_path):
        raise FileNotFoundError(f"Checkpoint {cp_path} not found.")
    with open(cp_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_agent(
    user_message,
    scenario_id,
    run_id=None,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tool_step=None
):
    start_time = time.monotonic()
    if run_id is None:
        run_id = f"run_{scenario_id}_{int(time.time()*1000)}"

    messages = [
        {"role": "system", "content": "AGENT. You are a refund assistant. Check order details, consult refund policy, and decide whether to issue or escalate a refund."},
        {"role": "user", "content": user_message}
    ]

    tokens_used = 0
    llm_calls = 0
    steps = []
    tools_executed = []

    for it in range(max_iterations):
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            return {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK: Execution timed out. Handing over to human support agent with full ticket history.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }

        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens

        # Check token limit immediately after LLM call, before executing requested tool
        if tokens_used > token_limit:
            return {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK: Token budget limit exceeded. Handing over case summary to human support lead.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }

        msg = resp.choices[0].message

        # If model returned no tool calls, it has finished with a final answer
        if not msg.tool_calls:
            ans = msg.content
            status = ans.split(":")[0].strip() if ":" in ans else ans.strip()
            return {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": status,
                "final_answer": ans,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }

        # Model requested a tool execution
        tc = msg.tool_calls[0]
        tool_name = tc.name
        tool_args = tc.arguments

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc.id, "name": tool_name, "arguments": tool_args}]
        })

        # Execute tool
        func = TOOL_MAP[tool_name]
        obs = func(**tool_args)

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tool_name,
            "content": json.dumps(obs)
        })

        tools_executed.append(tool_name)
        steps.append({
            "name": tool_name,
            "arguments": tool_args,
            "observation": obs
        })

        # Save checkpoint after every tool observation is appended
        save_checkpoint(run_id, scenario_id, messages, tokens_used, len(steps), tools_executed, steps)

        # Simulated crash if requested
        if crash_after_tool_step and len(steps) == crash_after_tool_step:
            raise RuntimeError(f"Simulated system crash after tool execution step {len(steps)} ({tool_name})")

    # If max_iterations reached without final answer
    return {
        "scenario_id": scenario_id,
        "run_id": run_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "FALLBACK: Maximum agent steps reached without resolution. Routing to human support team with context summary.",
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "tools_executed": tools_executed
    }

def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):
    start_time = time.monotonic()
    cp = load_checkpoint(run_id)

    scenario_id = cp["scenario_id"]
    messages = cp["messages"]
    tokens_used = cp["tokens_used"]
    tools_executed = list(cp["tools_executed"])
    steps = list(cp["steps"])
    llm_calls = 0

    for it in range(len(tools_executed), max_iterations):
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            return {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK: Execution timed out during resume. Routing to human agent.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }

        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens

        if tokens_used > token_limit:
            return {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK: Token limit exceeded during resume. Routing to human agent.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }

        msg = resp.choices[0].message
        if not msg.tool_calls:
            ans = msg.content
            status = ans.split(":")[0].strip() if ":" in ans else ans.strip()
            return {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": status,
                "final_answer": ans,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }

        tc = msg.tool_calls[0]
        tool_name = tc.name
        tool_args = tc.arguments

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc.id, "name": tool_name, "arguments": tool_args}]
        })

        func = TOOL_MAP[tool_name]
        obs = func(**tool_args)

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tool_name,
            "content": json.dumps(obs)
        })

        tools_executed.append(tool_name)
        steps.append({
            "name": tool_name,
            "arguments": tool_args,
            "observation": obs
        })

        save_checkpoint(run_id, scenario_id, messages, tokens_used, len(steps), tools_executed, steps)

    return {
        "scenario_id": scenario_id,
        "run_id": run_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "FALLBACK: Maximum iterations reached during resume.",
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "tools_executed": tools_executed
    }

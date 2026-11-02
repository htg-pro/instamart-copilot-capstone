"""Bounded ReAct execution with atomic checkpoints and local replay protection."""
import json
import os
from pathlib import Path
import re
import time
from uuid import uuid4

import mock_llm
from part_d.tools import TOOLS_SPEC, execute_tool

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "state"


class SimulatedCrash(RuntimeError):
    """Intentional failure after a saved tool observation for the recovery exercise."""


def _checkpoint_path(run_id, state_dir=STATE_DIR):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) or run_id in (".", ".."):
        raise ValueError("run_id must be a simple filename containing letters, numbers, dots, underscores or hyphens.")
    return Path(state_dir) / f"{run_id}.json"


def save_checkpoint(checkpoint, state_dir=STATE_DIR):
    """Flush a complete snapshot and atomically replace this run's checkpoint file."""
    path = _checkpoint_path(checkpoint["run_id"], state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(checkpoint, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_checkpoint(run_id, state_dir=STATE_DIR):
    return json.loads(_checkpoint_path(run_id, state_dir).read_text(encoding="utf-8"))


def _run_loop(state, max_iterations, token_limit, time_limit_seconds,
              crash_after_tool_step=None, state_dir=STATE_DIR):
    started = time.monotonic()
    previous_elapsed = state.get("elapsed_seconds", 0.0)
    calls_before_resume = state["llm_calls"]

    def checkpoint():
        state["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
        state["step_number"] = len(state["steps"])
        state["limits"] = {"max_iterations": max_iterations, "token_limit": token_limit,
                           "time_limit_seconds": time_limit_seconds}
        save_checkpoint(state, state_dir)

    def finish(status, answer):
        result = {key: state[key] for key in (
            "run_id", "scenario_id", "steps", "llm_calls", "tokens_used", "tools_executed")}
        result.update({"status": status, "final_answer": answer})
        if calls_before_resume:
            result["resumed_llm_calls"] = state["llm_calls"] - calls_before_resume
        state["result"] = result
        checkpoint()
        return result

    def handoff(status, reason):
        completed = ", ".join(state["tools_executed"]) or "no tools"
        return finish(status, f"FALLBACK: {reason} Human support should review scenario "
                      f"{state['scenario_id']}; completed {completed}. "
                      f"Calls={state['llm_calls']}, tokens={state['tokens_used']}. "
                      "Full observations are retained in the checkpoint.")

    # An initial snapshot also records a zero-call timeout or a pre-tool token stop.
    checkpoint()
    while True:
        if previous_elapsed + time.monotonic() - started >= time_limit_seconds:
            return handoff("FALLBACK_TIMEOUT", "Wall-clock time limit reached.")
        if state["tokens_used"] > token_limit:
            return handoff("FALLBACK_TOKEN_LIMIT", "Token budget exceeded.")

        # Normally each reply has one call. Keeping remaining calls in the checkpoint
        # also permits resuming between observations of a multi-tool reply.
        if state["pending_tool_calls"]:
            tool_call = state["pending_tool_calls"][0]
            try:
                observation = execute_tool(tool_call["name"], tool_call["arguments"],
                                           state["run_id"], len(state["steps"]) + 1)
            except (KeyError, TypeError, ValueError) as error:
                return handoff("FALLBACK_TOOL_ERROR", f"Tool request failed validation: {error}.")
            state["messages"].append({"role": "tool", "tool_call_id": tool_call["id"],
                                      "name": tool_call["name"], "content": json.dumps(observation)})
            state["tools_executed"].append(tool_call["name"])
            state["steps"].append({"name": tool_call["name"], "arguments": tool_call["arguments"],
                                   "observation": observation})
            state["pending_tool_calls"].pop(0)
            checkpoint()  # Save AFTER the observation and BEFORE the simulated crash.
            if crash_after_tool_step is not None and len(state["steps"]) == crash_after_tool_step:
                raise SimulatedCrash(f"Simulated crash after saved tool step {len(state['steps'])} "
                                     f"({tool_call['name']}).")
            continue

        if state["llm_calls"] >= max_iterations:
            return handoff("FALLBACK_MAX_ITERATIONS", "Maximum model-call iterations reached without resolution.")

        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += int(response.usage.total_tokens)
        if state["tokens_used"] > token_limit:
            # The newly requested tool is deliberately never executed.
            return handoff("FALLBACK_TOKEN_LIMIT", "Token budget exceeded after the latest model call.")

        reply = response.choices[0].message
        if not reply.tool_calls:
            answer = reply.content or ""
            state["messages"].append({"role": "assistant", "content": answer})
            return finish(answer.split(":", 1)[0].strip(), answer)

        calls = [{"id": call.id, "name": call.name, "arguments": call.arguments}
                 for call in reply.tool_calls]
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": calls})
        state["pending_tool_calls"] = list(calls)


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, *, run_id=None, crash_after_tool_step=None,
              state_dir=STATE_DIR):
    """Run the refund workflow, retaining every observation and enforcing all limits."""
    if max_iterations < 0 or token_limit < 0 or time_limit_seconds < 0:
        raise ValueError("Iteration, token, and time limits must be nonnegative.")
    run_id = run_id or f"run_{scenario_id}_{uuid4().hex[:12]}"
    state = {"run_id": run_id, "scenario_id": scenario_id,
             "messages": [
                 {"role": "system", "content": "AGENT. Look up the order, consult policy, calculate the refund, "
                  "and issue or escalate it. Only completed orders qualify; delivery must exceed 45 minutes. "
                  "Refunds above INR 1000 require a human. All requests are within seven days."},
                 {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "step_number": 0, "steps": [],
             "tools_executed": [], "pending_tool_calls": [], "elapsed_seconds": 0.0}
    return _run_loop(state, max_iterations, token_limit, time_limit_seconds,
                     crash_after_tool_step=crash_after_tool_step, state_dir=state_dir)


def resume_agent(run_id, max_iterations=None, token_limit=None, time_limit_seconds=None,
                 *, state_dir=STATE_DIR):
    """Continue a saved run using cumulative model calls, tokens, and active elapsed time."""
    state = load_checkpoint(run_id, state_dir)
    if "result" in state:
        return state["result"]  # A finished checkpoint cannot issue a second refund.
    # Compatibility with earlier checkpoints that contained one tool per model call.
    state.setdefault("llm_calls", state.get("step_number", len(state["steps"])))
    state.setdefault("pending_tool_calls", [])
    limits = state.get("limits", {})
    return _run_loop(
        state,
        limits.get("max_iterations", 6) if max_iterations is None else max_iterations,
        limits.get("token_limit", 6000) if token_limit is None else token_limit,
        limits.get("time_limit_seconds", 10) if time_limit_seconds is None else time_limit_seconds,
        state_dir=state_dir,
    )

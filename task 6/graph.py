import math
from operator import add
import re
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

import agents


MAX_DECISIONS = 8


class State(TypedDict, total=False):
    dialogue: Annotated[list[dict], add]
    execution_results: Annotated[list[dict], add]
    decision: dict
    decision_count: int
    test_rmse: float | None  # Volts, parsed from actual script stdout.
    stop_reason: str


# Recognize a printed measurement, not keywords used to choose an agent.
# Examples: Test RMSE: 0.007 V; Test free-run RMSE (mV): 7.0.
TEST_RMSE_PATTERN = re.compile(
    r"\btest\s+(?:free[- ]run\s+)?RMSE"
    r"(?:\s*\((?P<label_unit>mV|V)\))?\s*[:=]\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*(?P<unit>mV|V)?(?=$|\s|[,;])",
    re.IGNORECASE,
)


def _printed_test_rmse(state: dict) -> float | None:
    """Read the latest valid test RMSE from successful executor tool output."""
    for result in reversed(state.get("execution_results", [])):
        if result["exit_code"] != 0 or result.get("timed_out", False):
            continue
        matches = list(TEST_RMSE_PATTERN.finditer(result["stdout"]))
        for match in reversed(matches):
            unit = match.group("unit") or match.group("label_unit")
            value = float(match.group("value"))
            if unit and math.isfinite(value) and value >= 0:
                return value / 1000 if unit.lower() == "mv" else value
    return None


def build_graph(llm: BaseChatModel, coding_tool: BaseTool):
    """Connect the four roles; only the executor receives the coding tool."""

    def orchestrator_node(state: State) -> dict:
        # Also avoid a ninth call if an already-stopped state is invoked again.
        if state.get("decision_count", 0) >= MAX_DECISIONS:
            return {"stop_reason": "decision_limit"}

        update = agents.orchestrator(state, llm)
        rmse = _printed_test_rmse(state)
        next_role = update["decision"]["next_role"]
        update["test_rmse"] = rmse
        update["stop_reason"] = "running"

        if next_role == "finish" and rmse is not None:
            update["stop_reason"] = "finished"
        elif update["decision_count"] >= MAX_DECISIONS:
            # Stop immediately after decision 8, without dispatching another role.
            update["stop_reason"] = "decision_limit"
            update["dialogue"][0]["text"] += "\nThe eight-decision limit has been reached."
        elif next_role == "finish":
            # Reject premature finish, then let the LLM choose again.
            update["dialogue"][0]["text"] += (
                "\nFinish blocked: no successful script run has printed a finite "
                "test RMSE with units in stdout. Choose the next role yourself. "
                "An executor script should print: Test RMSE: <number> V."
            )
        return update

    def route(state: State) -> str:
        if state["stop_reason"] != "running":
            return END
        next_role = state["decision"]["next_role"]
        return "orchestrator" if next_role == "finish" else next_role

    builder = StateGraph(State)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("scientist", lambda state: agents.scientist(state, llm))
    builder.add_node("general", lambda state: agents.general(state, llm))
    builder.add_node("executor", lambda state: agents.executor(state, llm, coding_tool))

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", route, {
        "orchestrator": "orchestrator", "scientist": "scientist",
        "general": "general", "executor": "executor", END: END,
    })
    for role in ("scientist", "general", "executor"):
        builder.add_edge(role, "orchestrator")

    return builder.compile()

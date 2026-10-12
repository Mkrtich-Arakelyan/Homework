import math
from operator import add
import re
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

import agents


MAX_DECISIONS = 8


class State(TypedDict, total=False):
    dialogue: Annotated[list[dict], add]
    execution_results: Annotated[list[dict], add]
    decision: dict
    decision_count: int
    approved_decision_count: int | None  # Approval applies only to this decision.
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
    """Connect the four agents and a human pause/resume destination."""

    def orchestrator_node(state: State) -> dict:
        # Also avoid a ninth call if an already-stopped state is invoked again.
        if state.get("decision_count", 0) >= MAX_DECISIONS:
            return {"stop_reason": "decision_limit"}

        update = agents.orchestrator(state, llm)
        rmse = _printed_test_rmse(state)
        next_role = update["decision"]["next_role"]
        update["test_rmse"] = rmse
        update["stop_reason"] = "running"
        update["approved_decision_count"] = None

        if next_role in ("executor", "finish"):
            update["dialogue"][0]["text"] += (
                f"\n{next_role.capitalize()} blocked: this new decision has no human approval. "
                "Choose human with human_kind=approval, identify the specific action in "
                "instruction, set proposed_role to the intended destination, and ask "
                "whether to proceed. No coding tool or finish action was invoked."
            )
            if next_role == "finish" and rmse is None:
                update["dialogue"][0]["text"] += (
                    "\nFinish also requires a successful script run that printed a finite, "
                    "nonnegative test RMSE with units. Request approval to obtain this "
                    "evidence if it is missing."
                )

        if update["decision_count"] >= MAX_DECISIONS and next_role != "human":
            # A question at decision 8 may still receive and record its answer.
            # Other specialist work stops at the original eight-decision limit.
            update["stop_reason"] = "decision_limit"
            update["dialogue"][0]["text"] += "\nThe eight-decision limit has been reached."
        return update

    def executor_node(state: State) -> dict:
        decision = state.get("decision", {})
        approved_count = state.get("approved_decision_count")
        if not (
            decision.get("next_role") == "human"
            and decision.get("human_kind") == "approval"
            and decision.get("proposed_role") == "executor"
            and approved_count is not None
            and approved_count == state.get("decision_count")
        ):
            return {
                "approved_decision_count": None,
                "dialogue": [{
                    "role": "system",
                    "text": (
                        "Executor blocked: this saved proposal has no matching human "
                        "approval. No coding tool was invoked."
                    ),
                }],
            }

        update = agents.executor(state, llm, coding_tool)
        # A completed executor turn consumes approval, including failed attempts
        # and turns that return without running a script.
        update["approved_decision_count"] = None
        return update

    def human_node(state: State) -> dict:
        decision = state["decision"]
        # This node restarts on resume. Keep all side effects after interrupt().
        response = interrupt({
            "question": decision["question"],
            "human_kind": decision["human_kind"],
            "proposed_role": decision["proposed_role"],
            "instruction": decision["instruction"],
            "decision_count": state["decision_count"],
        })

        # Clarifications may resume with plain text. Approvals use an explicit
        # boolean plus the exact typed text; the CLI will collect both.
        if isinstance(response, str) and decision["human_kind"] == "clarification":
            reply = response
        elif isinstance(response, dict) and isinstance(response.get("text"), str):
            reply = response["text"]
        else:
            raise ValueError("Resume with clarification text or a dict containing text.")

        update = {
            "dialogue": [{"role": "human", "text": reply}],
            "approved_decision_count": None,
            "stop_reason": "running",
        }
        if decision["human_kind"] == "approval":
            if not isinstance(response, dict) or type(response.get("approved")) is not bool:
                raise ValueError("Approval replies require text and an explicit approved boolean.")
            if not response["approved"]:
                update["dialogue"].append({
                    "role": "system",
                    "text": (
                        f"Human refused the proposed {decision['proposed_role']} action: "
                        f"{decision['instruction']}\n"
                        "Action skipped; no coding tool or finish action was invoked "
                        "for this proposal."
                    ),
                })
            elif decision["proposed_role"] == "executor":
                update["approved_decision_count"] = state["decision_count"]
            elif decision["proposed_role"] == "finish":
                rmse = _printed_test_rmse(state)
                update["test_rmse"] = rmse
                if rmse is not None:
                    update["stop_reason"] = "finished"
                else:
                    update["dialogue"].append({
                        "role": "system",
                        "text": (
                            "Approved finish skipped: no successful script has printed "
                            "a finite, nonnegative test RMSE with units."
                        ),
                    })

        if update["stop_reason"] != "finished" and state["decision_count"] >= MAX_DECISIONS:
            update["approved_decision_count"] = None
            update["stop_reason"] = "decision_limit"
            update["dialogue"].append({
                "role": "system",
                "text": (
                    "Human reply recorded. The eight-decision limit has been reached; "
                    "no further action or orchestrator decision will run."
                ),
            })
        return update

    def route_human(state: State) -> str:
        if state["stop_reason"] != "running":
            return END
        if (state["decision"]["proposed_role"] == "executor"
                and state.get("approved_decision_count") == state["decision_count"]):
            return "executor"
        return "orchestrator"

    def route(state: State) -> str:
        if state["stop_reason"] != "running":
            return END
        next_role = state["decision"]["next_role"]
        return "orchestrator" if next_role in ("executor", "finish") else next_role

    builder = StateGraph(State)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("scientist", lambda state: agents.scientist(state, llm))
    builder.add_node("general", lambda state: agents.general(state, llm))
    builder.add_node("executor", executor_node)
    builder.add_node("human", human_node)

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", route, {
        "orchestrator": "orchestrator", "scientist": "scientist",
        "general": "general", "executor": "executor", "human": "human", END: END,
    })
    builder.add_conditional_edges("human", route_human, {
        "orchestrator": "orchestrator", "executor": "executor", END: END,
    })
    for role in ("scientist", "general", "executor"):
        builder.add_edge(role, "orchestrator")

    return builder.compile(checkpointer=InMemorySaver())

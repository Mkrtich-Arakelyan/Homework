import json
import math
import traceback
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


ORCHESTRATOR_PROMPT = """
You are the orchestrator. The user communicates only with you.
Read the complete role-labelled dialogue and choose the next role based on
what is needed now. Do not follow a fixed sequence or keyword-routing table.
Return next_role, an instruction for that role, and a one-sentence reason.
Coordinate the scientist for optical theory and experiment design, the general
agent for supporting concepts, and the executor for experiments and code tasks.
Pursue improved measured focusing enhancement at the supplied camera target.
Choose experiment settings autonomously through the agents; do not ask the
user to supply parameters or approval. You write no theory or code and have
no tools. Obtain an applicable scientist plan before focusing experiments.
Only the executor can run tools. When a planned experiment is ready to run,
choose next_role=executor, never scientist or general. Do not ask the scientist
to execute, trigger, or queue an experiment; that role can only propose or explain.
Reuse the existing scientist plan. Request another scientist turn only for a
new scientific question, interpretation that affects the next experiment, or
a necessary revision to the design. A plan does not need to be restated before
every tool call. Keep your role instructions short.
There are at most eight orchestrator decisions, including a finish decision.
Use the supplied decision-budget metadata; do not estimate usage from the
number of experiments. Each orchestrator choice consumes one decision, including
choosing scientist or general. Specialist replies and tool calls do not consume
additional orchestrator decisions. For your turn, remaining decisions include
the choice you are making now. Plan useful comparisons that leave one for finish.
If a successful focusing result exists and only one decision remains, you MUST
choose next_role=finish and summarize the evidence. Finish earlier if another
planning/execution step would leave no decision for reporting. Without successful
focusing evidence, do not claim completion, even when the budget is exhausted.
Use recorded tool results to decide what to try next; do not invent outcomes
or follow a hard-coded sequence of experiment settings.
Choose finish only after at least one run_focusing_experiment result has
ok=true, finite measured enhancement, saved output paths, and no validation
error. Supporting code results and proposed experiments do not count.
Each focusing-tool result already includes five fresh final-validation captures.
Use its final_enhancement directly for comparison and reporting. Do not schedule
an extra experiment merely to obtain final validation; a new tool call starts
a new search rather than remeasuring the previously selected pattern.
The exact metric is:
enhancement = focus_mean / max((TL_mean + TR_mean + BL_mean + BR_mean) / 4, 1e-12).
The numerical floor applies to the average of the four means.
When finishing, identify the best successful experiment by its final validated
enhancement, giving its experiment ID, settings, enhancement, and output files.
Prefer comparisons with fixed focus/background regions; if those regions
differ, explain that the scores use different metrics. If no successful
focusing measurement is recorded, continue the work rather than claim completion.
"""

SCIENTIST_PROMPT = """
You are the scientist. Follow the latest orchestrator instruction using the
problem and previous turns in the dialogue. Own the theory and scientific plan.
Explain the fixed complex transmission matrix, binary amplitude-only DMD,
noisy intensity measurements, macropixels, and enhancement metric when useful.
The exact enhancement metric is:
enhancement = focus_mean / max((TL_mean + TR_mean + BL_mean + BR_mean) / 4, 1e-12).
The four region means have equal weight. The denominator is their average,
not the largest region mean. Measurements are noisy; distinguish expectations
from results.
Propose valid settings using the bench dimensions, target, and tool constraints
supplied in the dialogue. Consider macropixel size, focus size, background size,
iterations, and search seed; explain why the proposed settings are useful.
Give a short numbered plan with a brief justification. Use the tool's actual
parameters: macropixel_size, focus_size, background_size, iterations, search_seed.
Capture counts are fixed: three per search measurement and five for final
validation. Every tool call already returns a freshly validated final_enhancement;
do not propose a separate final-validation experiment or configurable shot counts.
Use the supplied remaining decision count to keep comparisons feasible and
leave one decision for finishing. Reuse the existing plan in later replies;
describe only changed settings or useful new interpretation, not the whole plan.
Use actual previous measurements when recommending subsequent experiments.
Prefer keeping focus/background regions fixed when comparing macropixel sizes
or iteration budgets, since changing these regions changes the metric.
Respect the remaining eight-decision budget. Do not invent measurements or
execute tools; the executor performs all experiments.
If asked to execute an experiment, briefly give its ready-to-run settings and
tell the orchestrator to choose executor next. Do not claim you have triggered
or queued a call, and do not ask the user whether to proceed.
"""

GENERAL_PROMPT = """
You are the general agent. Follow the latest orchestrator instruction.
Explain supporting optical and programming concepts clearly and concisely.
Do not take over experiment design, write executable code, or execute tools.
Do not invent measurements; refer only to actual recorded tool results.
"""

EXECUTOR_PROMPT = """
You are the executor, the only role with tools.
Follow the latest orchestrator instruction and the available scientist plan.
For optical experiments use run_focusing_experiment with the selected settings.
Reuse the available plan without requesting it again merely to restate settings.
If no scientist plan exists, request one before conducting a focusing experiment.
Do not recreate the bench or implement a competing search through write_and_run.
Use write_and_run only for supporting code tasks explicitly requested by the
orchestrator. Supply a complete script and a fresh filename for those tasks.
Make exactly one tool call per executor turn.
The tool fixes search measurements at three captures and final validation at
five fresh captures. Its returned final_enhancement is already validated; do
not request another experiment just for validation. Enhancement is exactly
focus_mean / max((TL_mean + TR_mean + BL_mean + BR_mean) / 4, 1e-12).
Report the actual returned result, including measured enhancements and saved
paths for successful focusing experiments. Do not invent results or confuse
expected improvement with a measured improvement. On failure, report the
returned error or coding traceback/timeout accurately. A result with a
validation_error is invalid evidence and must not be reported as a success.
After a tool result, give a brief summary without repeating the full plan or
requesting another tool call.
"""


class RoutingDecision(BaseModel):
    """Required output of each orchestrator call."""

    next_role: Literal["scientist", "general", "executor", "finish"]
    instruction: str = Field(min_length=1, description="Instruction for the chosen role")
    reason: str = Field(min_length=1, description="One-sentence reason for this choice")


def _messages(state: dict, prompt: str) -> list:
    """Keep agent names intact without pretending they are API message roles."""
    used = state.get("decision_count", 0)
    budget = {
        "orchestrator_decision_limit": 8,
        "orchestrator_decisions_used": used,
        "orchestrator_decisions_remaining": max(0, 8 - used),
    }
    dialogue = json.dumps(state["dialogue"], ensure_ascii=False)
    evidence = json.dumps(state.get("execution_results", []), ensure_ascii=False)
    return [SystemMessage(content=prompt), HumanMessage(
        content="Decision budget:\n" + json.dumps(budget)
        + "\n\nDialogue:\n" + dialogue + "\n\nRecorded tool results:\n" + evidence
    )]


def orchestrator(state: dict, llm: BaseChatModel) -> dict:
    decision = llm.with_structured_output(
        RoutingDecision, method="json_schema"
    ).invoke(_messages(state, ORCHESTRATOR_PROMPT))
    return {
        "dialogue": [{"role": "orchestrator", "text": decision.model_dump_json()}],
        "decision": decision.model_dump(),
        "decision_count": state.get("decision_count", 0) + 1,
    }


def scientist(state: dict, llm: BaseChatModel) -> dict:
    response = llm.invoke(_messages(state, SCIENTIST_PROMPT))
    return {"dialogue": [{"role": "scientist", "text": response.text}]}


def general(state: dict, llm: BaseChatModel) -> dict:
    response = llm.invoke(_messages(state, GENERAL_PROMPT))
    return {"dialogue": [{"role": "general", "text": response.text}]}


def executor(
    state: dict,
    llm: BaseChatModel,
    coding_tool: BaseTool,
    focusing_tool: BaseTool,
) -> dict:
    """Dispatch one requested tool call and retain its result as evidence."""
    tool_map = {coding_tool.name: coding_tool, focusing_tool.name: focusing_tool}
    available_tools = list(tool_map.values())
    messages = _messages(state, EXECUTOR_PROMPT)
    request = llm.bind_tools(available_tools, tool_choice="any").invoke(messages)
    if len(request.tool_calls) != 1:
        return {"dialogue": [{
            "role": "executor",
            "text": "No tool ran: expected exactly one tool call. Please retry."
        }]}

    call = request.tool_calls[0]
    if call["name"] not in tool_map:
        return {"dialogue": [{
            "role": "executor", "text": "No tool ran: the requested tool is unavailable."
        }]}
    is_focusing = call["name"] == focusing_tool.name
    if is_focusing and not any(turn["role"] == "scientist" for turn in state["dialogue"]):
        return {"dialogue": [{
            "role": "executor", "text": "No tool ran: I need a scientist plan before focusing."
        }]}

    try:
        result = tool_map[call["name"]].invoke(call["args"])
    except Exception as error:
        if is_focusing:
            result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        else:
            result = {"exit_code": None, "stdout": "", "stderr": traceback.format_exc()}

    validation_error = None
    try:
        if not isinstance(result, dict):
            raise ValueError("The tool must return a dictionary.")
        if not is_focusing:
            if not {"exit_code", "stdout", "stderr"} <= result.keys():
                raise ValueError("Coding tool must return exit_code, stdout, and stderr.")
        else:
            if not isinstance(result.get("ok"), bool):
                raise ValueError("Focusing tool must return a boolean ok.")
            if result["ok"]:
                required = {
                    "experiment_id", "settings", "initial_enhancement",
                    "final_enhancement", "paths",
                }
                if not required <= result.keys():
                    raise ValueError("Successful focusing result is missing required fields.")
                if not isinstance(result["experiment_id"], str) or not result["experiment_id"]:
                    raise ValueError("A successful experiment needs an experiment ID.")
                if not isinstance(result["settings"], dict) or not result["settings"]:
                    raise ValueError("A successful experiment needs its settings.")
                paths = result["paths"]
                if (
                    not isinstance(paths, dict)
                    or not {"initial_camera", "final_camera", "final_pattern",
                            "results_json", "experiments_csv"} <= paths.keys()
                    or not all(isinstance(path, str) and path for path in paths.values())
                ):
                    raise ValueError("A successful experiment needs saved output paths.")
                for name in ("initial_enhancement", "final_enhancement"):
                    value = result[name]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                    ):
                        raise ValueError(f"{name} must be a finite number.")
            elif not isinstance(result.get("error"), str) or not result["error"]:
                raise ValueError("A failed focusing result needs an error message.")
    except (ValueError, OverflowError) as error:
        validation_error = str(error)

    recorded = {"tool_name": call["name"], "tool_args": call["args"], "result": result}
    if validation_error:
        recorded["validation_error"] = validation_error
    evidence = json.dumps(result, ensure_ascii=False)
    # Preserve the original AIMessage, including its encrypted reasoning content.
    tool_message = evidence
    if validation_error:
        tool_message += "\nResult validation error: " + validation_error
    messages.extend([request, ToolMessage(content=tool_message, tool_call_id=call["id"])])
    try:
        # Keep both tool declarations, but disable further calls.
        response = llm.bind_tools(available_tools, tool_choice="none").invoke(messages)
        summary = response.text
    except Exception as error:
        # A summary failure must not discard a completed tool result.
        summary = f"Model summary unavailable ({type(error).__name__}); see the raw tool result."
    if validation_error:
        summary += "\nResult validation error: " + validation_error
    return {
        "dialogue": [{
            "role": "executor",
            "text": summary + f"\n\nTool: {call['name']}\nRaw tool result:\n" + evidence,
        }],
        "execution_results": [recorded],
    }

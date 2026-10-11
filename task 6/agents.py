import json
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
Choose scientist for the equations or fitting plan, general for supporting
conceptual questions, and executor for implementation, execution, or a traceback.
You write no scientific theory and no code, and have no coding tool.
There are at most eight orchestrator decisions, including a finish decision.
Choose finish only after an executor turn contains a numerical test RMSE
printed by a Python script that actually ran. Planned or guessed results do
not count. Ask the executor to print the result if evidence is missing.
"""

SCIENTIST_PROMPT = """
You are the scientist. Follow the latest orchestrator instruction using the
problem and previous turns in the dialogue. Own the theory and scientific plan.
Specify the differential equation, proposed neural network and scaling, state
equation to integrate, integration method, training procedure and loss, and
free-run evaluation. Finish with numbered steps specific enough to implement.
Respect the supplied Task 5 requirements, including its data preprocessing,
architecture comparisons, checkpoint selection, and uninterrupted free-run test.
Make the implementation practical within the five-minute limit per script run;
do not omit required work silently. You have no coding tool and do not execute.
Never claim a measured number unless an executor turn already supplies that
number from real execution. Distinguish proposals from observed results.
"""

GENERAL_PROMPT = """
You are the general agent. Follow the latest orchestrator instruction.
Explain supporting high-level concepts: what a dataset record means, what a
term means, or what a reported error measures. Be clear and concise.
You do not design the fitting plan, write executable code, or debug tracebacks.
Those belong to the scientist and executor respectively. You have no coding tool.
Do not invent measurements; refer only to results already recorded by executor.
"""

EXECUTOR_PROMPT = """
You are the executor, the only role with a coding tool.
Follow the latest orchestrator instruction and implement the latest scientist
plan. If there is no scientist plan, ask for one without running code.
Write a complete Python script into a new .py file using the supplied tool.
Choose a fresh filename for each attempt. The tool writes and executes your
script in this run's working directory, with a five-minute timeout.
Call the tool exactly once per turn; later turns can repair failed scripts.
Use the dataset path and environment information supplied in the dialogue.
Print the training and test RMSE from the required free-run evaluation with
clear labels and units. Never print a guessed or previously recorded score
as a newly measured result. Report which file ran and its actual result.
If execution fails, report the tool's traceback or timeout accurately.
Repairs must preserve the scientist's differential equation and free-run test;
request a scientist revision if the scientific plan needs changing.
After a tool result, summarize it without requesting another tool call.
"""


class RoutingDecision(BaseModel):
    """Required output of each orchestrator call."""

    next_role: Literal["scientist", "general", "executor", "finish"]
    instruction: str = Field(min_length=1, description="Instruction for the chosen role")
    reason: str = Field(min_length=1, description="One-sentence reason for this choice")


def _messages(state: dict, prompt: str) -> list:
    """Keep agent names intact without pretending they are API message roles."""
    dialogue = json.dumps(state["dialogue"], ensure_ascii=False)
    return [SystemMessage(content=prompt), HumanMessage(content="Dialogue:\n" + dialogue)]


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


def executor(state: dict, llm: BaseChatModel, coding_tool: BaseTool) -> dict:
    """Generate a script, execute through the supplied tool, then report evidence."""
    if not any(turn["role"] == "scientist" for turn in state["dialogue"]):
        return {"dialogue": [{
            "role": "executor", "text": "I need a scientist plan before writing or running code."
        }]}

    messages = _messages(state, EXECUTOR_PROMPT)
    request = llm.bind_tools([coding_tool], tool_choice="any").invoke(messages)
    if len(request.tool_calls) != 1:
        return {"dialogue": [{
            "role": "executor",
            "text": "No script ran: expected exactly one coding-tool call. Please retry."
        }]}

    call = request.tool_calls[0]
    if call["name"] != coding_tool.name:
        return {"dialogue": [{
            "role": "executor", "text": "No script ran: the requested tool is unavailable."
        }]}

    try:
        # tools.py must return a dict with exit_code, stdout, and stderr.
        result = coding_tool.invoke(call["args"])
        if not isinstance(result, dict) or not {"exit_code", "stdout", "stderr"} <= result.keys():
            raise ValueError("Coding tool must return exit_code, stdout, and stderr.")
    except Exception:
        result = {"exit_code": None, "stdout": "", "stderr": traceback.format_exc()}

    evidence = json.dumps(result, ensure_ascii=False)
    # Preserve the original AIMessage, including Gemini's thought signatures.
    messages.extend([request, ToolMessage(content=evidence, tool_call_id=call["id"])])
    try:
        # Keep the tool declaration for Gemini, but disable further calls.
        response = llm.bind_tools([coding_tool], tool_choice="none").invoke(messages)
        summary = response.text
    except Exception as error:
        # A rate-limit/network failure must not discard a completed script run.
        summary = f"Model summary unavailable ({type(error).__name__}); see the raw tool result."
    return {
        "dialogue": [{
            "role": "executor", "text": summary + "\n\nRaw tool result:\n" + evidence
        }],
        "execution_results": [{"filename": call["args"].get("filename"), **result}],
    }

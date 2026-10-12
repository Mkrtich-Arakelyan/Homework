import json
import traceback
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator


ORCHESTRATOR_PROMPT = """
You are the orchestrator. The user communicates only with you.
Read the complete role-labelled dialogue and choose the next role based on
what is needed now. Do not follow a fixed sequence or keyword-routing table.
Return an object with one field, decision. Inside decision, return next_role,
instruction, reason, human_kind, question, and proposed_role. Choose one valid
shape: a non-human decision with all human fields null; a human clarification
with a nonblank question and proposed_role null; or a human approval with a
nonblank question and proposed_role executor or finish.
The reason must be one sentence. Choose scientist for equations or a fitting
plan and general for supporting conceptual questions. The executor handles
implementation, execution, and script repairs, subject to human approval.
You write no scientific theory and no code, and have no coding tool.
Treat the Silverbox background as context; follow the user's actual request.
Do not turn a question or checkpoint evaluation into full retraining.

Choose human for clarification when the latest request has multiple plausible
interpretations that would send work to different roles or change the reported
number. Set human_kind to clarification, ask which interpretation to follow,
and leave proposed_role as null. Use instruction to describe what needs resolving.
Resolve this ambiguity before proposing execution. On the next decision, use
the exact human reply in the dialogue to follow the selected interpretation.
Ask only for unresolved intent or missing essential information. Use details
already provided; resolve implementation questions through the scientist and
approved executor inspection rather than guessing.

Before proposing human approval for an executor action, ensure the dialogue
contains a scientist plan applicable to the requested work. If it does not,
choose scientist to provide that plan first, after resolving any ambiguity.
This also applies to evaluating a saved checkpoint or comparing it with a
baseline: request an evaluation plan without retraining. The task background,
your own instructions, and human approval do not replace a scientist plan.
If the executor reports that a scientist plan is missing, choose scientist
next; obtain fresh human approval for the resulting executor proposal.

Choose human for approval before running the coding tool, overwriting an
existing result on disk, or ending the dialogue and reporting a measurement.
Set human_kind to approval. Set proposed_role to executor for a coding action
or finish for final reporting. Put the specific proposed action in instruction
and ask whether to proceed in question. Identify the intended computation and
outputs, including any existing result that would be overwritten.
For these actions, choose human instead of directly choosing executor or finish.
Propose finish only when a successful executor script has printed a finite,
nonnegative test RMSE with units. Planned, guessed, or failed-run scores do not
count. If evidence is missing, propose an approved executor action to obtain it.

Human replies appear under role human. A clarification is not execution approval.
Acceptance applies only to the saved proposal; the graph dispatches that action.
On refusal, the proposed action must not run. Read the human's feedback and
choose an appropriate next step, such as revising the plan or explaining it.
Never treat silence, your own decision, or an earlier approval as permission
for a new or changed action.

Every human decision needs a nonblank question. For ordinary role decisions,
leave human_kind, question, and proposed_role as null.
There are at most eight orchestrator decisions, including the decision proposing
to finish. Human answers are not themselves orchestrator decisions.
"""

SCIENTIST_PROMPT = """
You are the scientist. Follow the latest orchestrator instruction using the
problem and previous turns, including human clarifications and refusals.
Own the theory and scientific plan. Answer the specific request; do not propose
full retraining when the user asks for an explanation or checkpoint evaluation.
For an existing model, plan evaluation using its saved architecture and
preprocessing. Identify any necessary executor inspection instead of guessing
missing details or designing a replacement model.
When a fitting plan is requested, specify the differential equation, proposed
neural network and scaling, state equation to integrate, integration method,
training procedure and loss, and free-run evaluation. Finish a plan with
numbered steps specific enough to implement.
Respect the supplied Task 5 requirements, including its data preprocessing,
architecture comparisons, checkpoint selection, and uninterrupted free-run test.
Make the implementation practical within the five-minute limit per script run;
do not omit required work silently. You have no coding tool and do not execute.
Never claim a measured number unless an executor turn already supplies that
number from real execution. Distinguish proposals from observed results.
"""

GENERAL_PROMPT = """
You are the general agent. Follow the latest orchestrator instruction.
Use human clarifications and refusals in the dialogue to answer the intended
question. Explain a proposed action if requested, without treating your
explanation as approval to perform it.
Explain supporting high-level concepts: what a dataset record means, what a
term means, or what a reported error measures. Be clear and concise.
You do not design the fitting plan, write executable code, or debug tracebacks.
Those belong to the scientist and executor respectively. You have no coding tool.
Do not invent measurements; refer only to results already recorded by executor.
"""

EXECUTOR_PROMPT = """
You are the executor, the only role with a coding tool.
Follow the saved executor proposal approved by the human and the latest
applicable scientist plan. Human approval applies only to that proposal;
do not expand evaluation into retraining, change the agreed computation, or
overwrite an existing result outside the approved scope. A clarification or
refusal is not approval. Authorization is enforced by the application, not by
your own assertion that permission exists.
If there is no scientist plan, ask for one without running code.
Write a complete Python script into a new .py file using the supplied tool.
Choose a fresh filename for each attempt. The tool writes and executes your
script in this run's working directory, with a five-minute timeout.
Call the tool exactly once per turn; later turns can repair failed scripts.
Use the dataset path and environment information supplied in the dialogue.
For an existing model, inspect its metadata and source as needed within the
approved scope. Preserve its architecture, weights, and preprocessing; load
weights strictly and report mismatches. Do not retrain unless requested.
Follow the evaluation and baseline definitions supplied in the background.
Print the training and test RMSE from the required free-run evaluation with
clear labels and units. Never print a guessed or previously recorded score
as a newly measured result. Report which file ran and its actual result.
If execution fails, report the tool's traceback or timeout accurately.
Repairs must preserve the scientist's differential equation and free-run test;
request a scientist revision if the scientific plan needs changing.
After a tool result, summarize it without requesting another tool call.
"""


class RoutingDecision(BaseModel):
    """Flat decision used by the graph, with local consistency checks."""

    model_config = ConfigDict(extra="forbid")

    next_role: Literal["scientist", "general", "executor", "human", "finish"]
    instruction: str = Field(min_length=1, description="Instruction for the chosen role")
    reason: str = Field(min_length=1, description="One-sentence reason for this choice")
    human_kind: Literal["clarification", "approval"] | None = Field(
        description="Whether the human should clarify the request or approve an action",
    )
    question: str | None = Field(
        min_length=1,
        description="The question to ask the human",
    )
    proposed_role: Literal["executor", "finish"] | None = Field(
        description="The action destination awaiting human approval",
    )

    @model_validator(mode="after")
    def validate_human_fields(self) -> "RoutingDecision":
        if self.next_role != "human":
            if any(value is not None for value in (
                self.human_kind, self.question, self.proposed_role
            )):
                raise ValueError("Non-human decisions must leave human fields as None.")
            return self

        if self.human_kind is None or self.question is None or not self.question.strip():
            raise ValueError("Human decisions require human_kind and a nonblank question.")
        if self.human_kind == "approval" and self.proposed_role is None:
            raise ValueError("Approval requests require proposed_role.")
        if self.human_kind == "clarification" and self.proposed_role is not None:
            raise ValueError("Clarification requests must leave proposed_role as None.")
        return self


class NonHumanDecision(RoutingDecision):
    """A role decision that cannot contain a human question or proposal."""

    next_role: Literal["scientist", "general", "executor", "finish"]
    human_kind: None
    question: None
    proposed_role: None


class ClarificationDecision(RoutingDecision):
    """Ask the human to resolve ambiguity without proposing an action."""

    next_role: Literal["human"]
    human_kind: Literal["clarification"]
    question: str = Field(min_length=1, pattern=r"\S", description="The clarification question")
    proposed_role: None


class ApprovalDecision(RoutingDecision):
    """Ask the human to approve a specific executor or finish action."""

    next_role: Literal["human"]
    human_kind: Literal["approval"]
    question: str = Field(min_length=1, pattern=r"\S", description="The approval question")
    proposed_role: Literal["executor", "finish"]


class RoutingOutput(BaseModel):
    """Nest the alternatives because OpenAI requires an object at the root."""

    model_config = ConfigDict(extra="forbid")

    decision: NonHumanDecision | ClarificationDecision | ApprovalDecision


def _messages(state: dict, prompt: str) -> list:
    """Keep agent names intact without pretending they are API message roles."""
    dialogue = json.dumps(state["dialogue"], ensure_ascii=False)
    return [SystemMessage(content=prompt), HumanMessage(content="Dialogue:\n" + dialogue)]


def orchestrator(state: dict, llm: BaseChatModel) -> dict:
    output = llm.with_structured_output(
        RoutingOutput, method="json_schema", strict=True
    ).invoke(_messages(state, ORCHESTRATOR_PROMPT))
    # Keep the graph and saved dialogue on their existing flat decision format.
    decision = RoutingDecision.model_validate(output.decision.model_dump())
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
    decision = state.get("decision", {})
    approved_count = state.get("approved_decision_count")
    if not (
        decision.get("next_role") == "human"
        and decision.get("human_kind") == "approval"
        and decision.get("proposed_role") == "executor"
        and approved_count is not None
        and approved_count == state.get("decision_count")
    ):
        return {"dialogue": [{
            "role": "executor",
            "text": "No script ran: this saved executor proposal requires matching human approval.",
        }]}

    if not any(turn["role"] == "scientist" for turn in state["dialogue"]):
        return {"dialogue": [{
            "role": "executor", "text": "I need a scientist plan before writing or running code."
        }]}

    messages = _messages(state, EXECUTOR_PROMPT)
    request = llm.bind_tools(
        [coding_tool], tool_choice="required", parallel_tool_calls=False, strict=True
    ).invoke(messages)
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
    # Preserve the original AIMessage, including OpenAI reasoning and call IDs.
    messages.extend([request, ToolMessage(content=evidence, tool_call_id=call["id"])])
    try:
        # Keep the tool declaration while disabling further calls.
        response = llm.bind_tools(
            [coding_tool], tool_choice="none", parallel_tool_calls=False, strict=True
        ).invoke(messages)
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

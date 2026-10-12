import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import traceback

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from graph import build_graph
from tools import create_coding_tool, RUN_TIMEOUT_SECONDS


BASE_DIR = Path(__file__).resolve().parent

# Background requirements; the user's actual request is supplied separately.
TASK5 = """
Task 5 background: identify the Silverbox system with a neural ODE in PyTorch.
The electronic circuit approximates the Duffing oscillator
    y'' + a*y' + b*y + c*y**3 = g*u(t).
Here a is damping, b and c are linear and cubic stiffness, and g is input gain;
these physical coefficients are distinct from neural-network weights.

Data: SNLS80mV.mat contains V1 (input u) and V2 (output y), sampled at
610.35 Hz. Use the 4096 samples at Python indices [42650:46746]. Subtract
each channel's mean over this entire excerpt. The first 3072 samples are
training data; the final 1024 are test data. The input is known during testing.

Replace acceleration with a multilayer fully connected neural network
alpha_theta(y, v, u). Integrate the state equations y'=v and
v'=alpha_theta(y,v,u) with classical RK4 and dt=1/610.35 seconds.
For requests to train or fit a new model, choose and explain network depth,
width, activation, and numerical scaling. Compare more than one architecture.
For each, tune optimizer, learning rate, number of training updates, and
rollout length; retain its best test checkpoint.

For an existing checkpoint, determine its architecture, weights, scaling, and
evaluation conventions from its metadata and associated source. Preserve the
saved model; do not invent missing details, substitute an architecture, or
retrain unless requested. Reference source may be inspected without running
its training routine. Any coding-tool inspection requires human approval.

For evaluation, freeze the weights and simulate one uninterrupted trajectory
over all 4096 samples, initialized from the first measured y and a finite
difference velocity. Drive it with measured u. Do not reset the state at the
train/test boundary or feed measured output back during the free run.
Compute train RMSE over the first 3072 samples and test RMSE over the final
1024 samples of that same trajectory, in volts.
Recompute errors from predictions and data; stored scores are historical
references, not newly measured results.

The zero-output baseline predicts y_hat[k] = 0 for every sample.
It requires no neural network or ODE simulation. It does not mean
setting acceleration to zero.

After subtracting the output mean over the entire 4096-sample
excerpt, compute:
baseline_train_rmse = sqrt(mean(y[:3072] ** 2))
baseline_test_rmse = sqrt(mean(y[3072:] ** 2))

Report both in volts. Compare the checkpoint's test RMSE against
baseline_test_rmse, using the same final 1024 samples.
Compute these values from the dataset; do not assume a score.

Save new artifacts in the current run directory using fresh filenames.
Print the baseline values as baseline_train_rmse_V and baseline_test_rmse_V.
Reserve Test RMSE: <number> V for the evaluated neural model, keeping its
measurement separate from the baseline.

Save the compared configurations and their measured test RMSEs, the retained
network and training settings, both retained RMSEs, and a plot of measured
versus predicted test output. These will support a later PDF describing the
state equations, method, comparisons, result, and GitHub link.
""".strip()


def save_run(state: dict, run_dir: Path) -> None:
    """Keep the latest state and a readable transcript after each graph turn."""
    (run_dir / "state.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    transcript = "\n\n".join(
        f"[{turn['role']}]\n{turn['text']}" for turn in state["dialogue"]
    )
    (run_dir / "dialogue.txt").write_text(transcript, encoding="utf-8")


def read_human_reply(question: dict) -> str | dict:
    """Collect the exact reply and validate approval before resuming the graph."""
    print(f"\n[human question]\n{question['question']}", flush=True)
    if question["human_kind"] == "clarification":
        return input("Your clarification: ")

    print(
        f"Proposed {question['proposed_role']} action: {question['instruction']}",
        flush=True,
    )
    while True:
        reply = input("Proceed? [yes/no, optional comment]: ")
        answer = re.match(r"^\s*(yes|y|no|n)(?=$|[\s,.:;!?])", reply, re.IGNORECASE)
        if answer:
            return {"text": reply, "approved": answer.group(1).lower() in ("yes", "y")}
        print("Please start your answer with yes or no.", flush=True)


def main() -> int:
    load_dotenv(BASE_DIR / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "SNLS80mV.mat")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-nano"))
    parser.add_argument("--request", help="Your Silverbox request; prompted if omitted.")
    args = parser.parse_args()
    data_path = args.data.expanduser().resolve()

    if not data_path.is_file():
        parser.error(f"Dataset not found: {data_path}. Use --data to specify its path.")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
        parser.error("Replace YOUR_OPENAI_API_KEY_HERE with your OPENAI_API_KEY in the .env beside main.py.")
    missing = [name for name in ("torch", "numpy", "scipy", "matplotlib")
               if importlib.util.find_spec(name) is None]
    if missing:
        parser.error("Install the script dependencies: " + ", ".join(missing))

    try:
        user_request = args.request if args.request is not None else input("Silverbox request: ")
    except (EOFError, KeyboardInterrupt):
        print("Request input cancelled.", file=sys.stderr)
        return 1
    if not user_request.strip():
        parser.error("Provide a nonblank request with --request or at the terminal prompt.")

    run_dir = BASE_DIR / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    config = {"configurable": {"thread_id": run_dir.name}}
    (run_dir / "config.json").write_text(json.dumps({
        "provider": "openai", "api": "responses", "reasoning_effort": "medium",
        "model": args.model, "dataset": str(data_path),
        "python": sys.version, "script_timeout_seconds": RUN_TIMEOUT_SECONDS,
        "thread_id": run_dir.name, "request": user_request,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    context = (
        "Silverbox background: apply these requirements to the requested work. "
        "Follow the separate user request; do not assume full retraining is required.\n\n"
        + TASK5 + "\n\nExecution environment:\n"
        f"Dataset absolute path: {data_path}\n"
        f"Run directory: {run_dir}\n"
        "Python with PyTorch, NumPy, SciPy, and Matplotlib is installed. Use CPU.\n"
        f"Each script must finish within {RUN_TIMEOUT_SECONDS} seconds.\n"
        "Save scripts and result artifacts in the run directory. Use a noninteractive "
        "Matplotlib backend and save plots instead of opening a window.\n"
        "Print the actual free-run result as: Test RMSE: <number> V.\n"
        "Work within eight orchestrator decisions, including the finish decision."
    )
    state = {"dialogue": [{"role": "system", "text": context},
                          {"role": "user", "text": user_request}],
             "execution_results": [], "decision_count": 0,
             "approved_decision_count": None,
             "test_rmse": None, "stop_reason": "running"}
    save_run(state, run_dir)
    print(f"Model: {args.model}\nResults: {run_dir}", flush=True)

    try:
        llm = ChatOpenAI(
            model=args.model, api_key=api_key, max_tokens=16384,
            timeout=120, max_retries=2,
            use_responses_api=True, output_version="responses/v1",
            reasoning={"effort": "medium"},
            # Replay the executor's reasoning with its tool result without
            # relying on a server-stored conversation.
            store=False, include=["reasoning.encrypted_content"],
        )
        workflow = build_graph(llm, create_coding_tool(run_dir))
        printed = len(state["dialogue"])
        stream_input = state
        while True:
            pending = ()
            for snapshot in workflow.stream(stream_input, config=config, stream_mode="values"):
                if "__interrupt__" in snapshot:
                    pending = snapshot["__interrupt__"]
                    continue
                state = snapshot
                save_run(state, run_dir)
                for turn in state["dialogue"][printed:]:
                    print(f"\n[{turn['role']}]\n{turn['text']}", flush=True)
                printed = len(state["dialogue"])

            if not pending:
                break
            reply = read_human_reply(pending[0].value)
            # Reuse the same thread; never resend the initial state on resume.
            stream_input = Command(resume=reply)
    except (Exception, KeyboardInterrupt):
        # Completed turns remain saved if the API fails or the user interrupts.
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Run interrupted. See {run_dir / 'error.txt'}", file=sys.stderr)
        return 1

    print(f"\nStopped: {state['stop_reason']}")
    if state["stop_reason"] == "finished" and state["test_rmse"] is not None:
        print(f"Recorded test RMSE: {state['test_rmse']:.6g} V")
    elif state["test_rmse"] is None:
        print("No successful script has reported a test RMSE.")
    else:
        print("Run ended without an approved final report.")
    print(f"Dialogue and execution evidence: {run_dir}")
    return 0 if state["stop_reason"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from graph import build_graph
from tools import create_coding_tool, RUN_TIMEOUT_SECONDS


BASE_DIR = Path(__file__).resolve().parent

# Supply the exercise, rather than our previous solution or its scores.
TASK5 = """
Solve Task 5: identify the Silverbox system with a neural ODE in PyTorch.
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
Choose and explain network depth, width, activation, and numerical scaling.
Compare more than one architecture. For each, tune optimizer, learning rate,
number of training updates, and rollout length; retain its best test checkpoint.

For evaluation, freeze the weights and simulate one uninterrupted trajectory
over all 4096 samples, initialized from the first measured y and a finite
difference velocity. Drive it with measured u. Do not reset the state at the
train/test boundary or feed measured output back during the free run.
Compute train RMSE over the first 3072 samples and test RMSE over the final
1024 samples of that same trajectory, in volts. Also compute the zero-output
baseline from the data; do not assume a score.

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


def main() -> int:
    load_dotenv(BASE_DIR / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "SNLS80mV.mat")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    args = parser.parse_args()
    data_path = args.data.expanduser().resolve()

    if not data_path.is_file():
        parser.error(f"Dataset not found: {data_path}. Use --data to specify its path.")
    if not os.getenv("GOOGLE_API_KEY", "").strip():
        parser.error("Set GOOGLE_API_KEY in the .env beside main.py.")
    missing = [name for name in ("torch", "numpy", "scipy", "matplotlib")
               if importlib.util.find_spec(name) is None]
    if missing:
        parser.error("Install the script dependencies: " + ", ".join(missing))

    run_dir = BASE_DIR / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({
        "model": args.model, "dataset": str(data_path),
        "python": sys.version, "script_timeout_seconds": RUN_TIMEOUT_SECONDS,
    }, indent=2), encoding="utf-8")

    request = (
        TASK5 + "\n\nExecution environment:\n"
        f"Dataset absolute path: {data_path}\n"
        f"Run directory: {run_dir}\n"
        "Python with PyTorch, NumPy, SciPy, and Matplotlib is installed. Use CPU.\n"
        f"Each script must finish within {RUN_TIMEOUT_SECONDS} seconds.\n"
        "Save scripts and result artifacts in the run directory. Use a noninteractive "
        "Matplotlib backend and save plots instead of opening a window.\n"
        "Print the actual free-run result as: Test RMSE: <number> V.\n"
        "Work within eight orchestrator decisions, including the finish decision."
    )
    state = {"dialogue": [{"role": "user", "text": request}],
             "execution_results": [], "decision_count": 0,
             "test_rmse": None, "stop_reason": "running"}
    save_run(state, run_dir)
    print(f"Model: {args.model}\nResults: {run_dir}", flush=True)

    try:
        llm = ChatGoogleGenerativeAI(
            model=args.model, vertexai=False, max_tokens=16384,
            timeout=120, max_retries=2,
        )
        workflow = build_graph(llm, create_coding_tool(run_dir))
        printed = len(state["dialogue"])
        for state in workflow.stream(state, stream_mode="values"):
            save_run(state, run_dir)
            for turn in state["dialogue"][printed:]:
                print(f"\n[{turn['role']}]\n{turn['text']}", flush=True)
            printed = len(state["dialogue"])
    except (Exception, KeyboardInterrupt):
        # Completed turns remain saved if the API fails or the user interrupts.
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Run interrupted. See {run_dir / 'error.txt'}", file=sys.stderr)
        return 1

    print(f"\nStopped: {state['stop_reason']}")
    if state["test_rmse"] is not None:
        print(f"Recorded test RMSE: {state['test_rmse']:.6g} V")
    else:
        print("No successful script has reported a test RMSE.")
    print(f"Dialogue and execution evidence: {run_dir}")
    return 0 if state["stop_reason"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())

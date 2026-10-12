"""Run Homework 2 Task 2: autonomous focusing on a simulated optical bench."""

import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REQUEST = (
    "Improve focusing at the fixed camera target. Autonomously choose and "
    "justify valid experiment settings, compare actual measured results, and "
    "report the best observed experiment with its enhancement, settings, "
    "and saved initial and final camera images."
)


def save_run(state: dict, run_dir: Path) -> None:
    """Save the latest graph state and a role-labelled transcript."""
    (run_dir / "state.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    transcript = "\n\n".join(
        f"[{turn['role']}]\n{turn['text']}" for turn in state["dialogue"]
    )
    if state["execution_results"]:
        transcript += "\n\n[tool executions]\n" + json.dumps(
            state["execution_results"], indent=2, ensure_ascii=False, allow_nan=False
        )
    (run_dir / "dialogue.txt").write_text(transcript, encoding="utf-8")


def main() -> int:
    load_dotenv(BASE_DIR / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-nano"))
    parser.add_argument("--request", default=DEFAULT_REQUEST, help="Override the autonomous request.")
    args = parser.parse_args()
    if not args.request.strip():
        parser.error("--request must not be blank.")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
        parser.error("Set OPENAI_API_KEY in your environment or the .env beside main.py.")
    missing = [
        name for name in ("numpy", "matplotlib", "langchain_openai", "langgraph")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        parser.error("Install the required dependencies: " + ", ".join(missing))

    run_dir = BASE_DIR / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    state = {
        "dialogue": [{"role": "user", "text": args.request}],
        "execution_results": [],
        "decision_count": 0,
        "stop_reason": "running",
    }
    print(f"Model: {args.model}\nResults: {run_dir}\nRequest: {args.request}", flush=True)

    try:
        from langchain_openai import ChatOpenAI
        from bench import SimulatedBench, SEARCH_SHOTS, VALIDATION_SHOTS
        from graph import build_graph, MAX_DECISIONS
        from tools import create_coding_tool, create_focusing_tool, RUN_TIMEOUT_SECONDS

        bench = SimulatedBench(
            dmd_shape=(16, 16),
            camera_shape=(32, 32),
            transmission_seed=42,
            noise_seed=43,
            noise_std=0.01,
        )
        bench_settings = {
            "dmd_shape": bench.dmd_shape,
            "camera_shape": bench.camera_shape,
            "target_position": bench.target_position,
            "transmission_seed": bench.transmission_seed,
            "noise_seed": bench.noise_seed,
            "noise_std": bench.noise_std,
        }
        context = (
            "Homework 2 Task 2: autonomous focusing through a simulated optical bench.\n"
            f"Bench settings: {json.dumps(bench_settings)}\n"
            "These dimensions and noise settings are simulation choices, not hardware specifications.\n"
            "The DMD controls binary amplitudes 0 or 1; an illuminated pattern cannot be all off.\n"
            "A fixed complex transmission matrix maps 256 DMD pixels to 1024 camera pixels.\n"
            "Camera intensity is I = abs(T @ pattern.ravel()) ** 2, followed by fresh "
            "additive Gaussian noise and clipping negative measurements to zero. "
            "Intensities are relative simulation units.\n"
            "One shared bench is used for the whole run: its matrix and target stay fixed, "
            "and its independent noise generator advances between captures.\n"
            "Use run_focusing_experiment for all optical experiments. Choose its parameters "
            "autonomously; do not recreate the bench or implement another search.\n"
            "Valid macropixel sizes are 1, 2, 4, and 8: each divides both DMD dimensions "
            "and leaves at least two controllable blocks. Each iteration flips one block.\n"
            "Focus and background sizes must be positive odd integers, with background "
            "larger than focus. Both centred squares must fit around the fixed target; "
            "the four background regions must be nonempty. Coordinates are (row, column).\n"
            "Background is the larger square outside the focus square, divided into "
            "top-left, top-right, bottom-left, and bottom-right regions. The target row "
            "belongs to bottom regions and the target column to right regions.\n"
            "Enhancement = focus mean / max(mean of the four background-region means, 1e-12). "
            "The four means have equal weight, regardless of region pixel counts.\n"
            f"Average {SEARCH_SHOTS} fresh captures for each search measurement and "
            f"{VALIDATION_SHOTS} fresh captures for final validation. The returned final "
            "enhancement comes from validation, not the highest noisy search score.\n"
            "Use 1 to 1,000 iterations per tool call. Improvement is not guaranteed.\n"
            "Keeping focus and background regions fixed makes comparisons between "
            "macropixel sizes or iteration budgets easier to interpret. Changing the "
            "regions changes the metric; acknowledge that when comparing results.\n"
            f"There are at most {MAX_DECISIONS} orchestrator decisions, including finish. "
            "Plan useful comparisons early enough to leave a decision for finishing.\n"
            f"Output directory: {run_dir}\n"
            "The focusing tool saves initial/final camera images, the final pattern, "
            "results.json, and an experiments.csv table. Do not duplicate these outputs.\n"
            "Use write_and_run only for explicitly needed supporting code tasks. "
            f"Such scripts must finish within {RUN_TIMEOUT_SECONDS} seconds.\n"
            "Base all conclusions on recorded tool results, never proposals or invented "
            "measurements. Finish only with successful focusing-tool evidence; identify "
            "the best observed experiment, settings, final enhancement, and saved paths."
        )
        state["dialogue"].insert(0, {"role": "system", "text": context})
        (run_dir / "config.json").write_text(json.dumps({
            "provider": "openai",
            "api": "responses",
            "model": args.model,
            "reasoning_effort": "medium",
            "python": sys.version,
            "bench": bench_settings,
            "request": args.request,
            "max_orchestrator_decisions": MAX_DECISIONS,
            "script_timeout_seconds": RUN_TIMEOUT_SECONDS,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        save_run(state, run_dir)

        llm = ChatOpenAI(
            model=args.model, api_key=api_key, max_tokens=16384,
            timeout=120, max_retries=2,
            use_responses_api=True, output_version="responses/v1",
            reasoning={"effort": "medium"},
            store=False, include=["reasoning.encrypted_content"],
        )
        coding_tool = create_coding_tool(run_dir)
        focusing_tool = create_focusing_tool(run_dir, bench)
        workflow = build_graph(llm, coding_tool, focusing_tool)
        printed_dialogue = len(state["dialogue"])
        printed_results = 0
        for snapshot in workflow.stream(state, stream_mode="values"):
            state = snapshot
            save_run(state, run_dir)
            for turn in state["dialogue"][printed_dialogue:]:
                print(f"\n[{turn['role']}]\n{turn['text']}", flush=True)
            for record in state["execution_results"][printed_results:]:
                print("\n[tool execution]\n" + json.dumps(
                    record, indent=2, ensure_ascii=False
                ), flush=True)
            printed_dialogue = len(state["dialogue"])
            printed_results = len(state["execution_results"])
    except (Exception, KeyboardInterrupt) as error:
        failure_trace = traceback.format_exc()
        state["stop_reason"] = "execution_failure"
        message = f"Execution/API failure ({type(error).__name__}); see error.txt."
        state["dialogue"].append({"role": "system", "text": message})
        try:
            (run_dir / "error.txt").write_text(failure_trace, encoding="utf-8")
            save_run(state, run_dir)
        except Exception as save_error:
            print(f"Could not save the latest failure evidence: {save_error}", file=sys.stderr)
        print(f"\n{message}\nLatest available evidence: {run_dir}", file=sys.stderr)
        return 1

    print(f"\nStopped: {state['stop_reason']}")
    if state["stop_reason"] == "finished":
        print("Finished with successful focusing-tool evidence.")
    elif state["stop_reason"] == "decision_limit":
        print("Decision budget reached; the workflow did not finish successfully.")
    else:
        print("The workflow ended without an evidence-backed finish.")

    successful = [
        record["result"] for record in state["execution_results"]
        if record.get("tool_name") == "run_focusing_experiment"
        and not record.get("validation_error")
        and isinstance(record.get("result"), dict)
        and record["result"].get("ok") is True
    ]
    if successful:
        best = max(successful, key=lambda result: result["final_enhancement"])
        print(f"Highest recorded final enhancement: {best['final_enhancement']:.6g}")
        print(f"Experiment: {best['experiment_id']}")
        print("Settings: " + json.dumps(best["settings"]))
        print("Saved outputs: " + json.dumps(best["paths"]))
    print(f"Dialogue and execution evidence: {run_dir}")
    return 0 if state["stop_reason"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())

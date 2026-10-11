from pathlib import Path
import re
import subprocess
import sys
import traceback

from langchain_core.tools import BaseTool, tool


RUN_TIMEOUT_SECONDS = 300


def _text(output: bytes | None) -> str:
    """Decode captured output, including partial output after a timeout."""
    return (output or b"").decode("utf-8", errors="replace")


def create_coding_tool(work_dir: str | Path) -> BaseTool:
    """Create a write_and_run tool tied to this run's working directory."""
    directory = Path(work_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    @tool
    def write_and_run(filename: str, code: str) -> dict:
        """Write a new Python file and execute it with a five-minute timeout.

        filename: A fresh filename such as attempt_1.py. Use only letters,
            digits, underscores, or hyphens before .py; do not supply a path.
        code: The complete Python source to write and run.

        Returns exit_code, stdout, stderr, and timed_out. An exit_code of None
        means there was no normal process exit code, such as on a timeout or
        file-writing failure. The script runs in the configured run directory
        using the same Python environment as the agent system.
        """
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.py", filename):
            return {
                "exit_code": None, "stdout": "", "timed_out": False,
                "stderr": "Invalid filename. Use a simple name such as attempt_1.py, without a path.",
            }

        script = directory / filename
        try:
            # Exclusive creation preserves earlier attempts and prevents overwriting.
            with script.open("x", encoding="utf-8") as file:
                file.write(code)

            run = subprocess.run(
                [sys.executable, "-X", "utf8", "-u", str(script)],
                cwd=directory,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=RUN_TIMEOUT_SECONDS,
            )
            return {
                "exit_code": run.returncode,
                "stdout": _text(run.stdout),
                "stderr": _text(run.stderr),
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as error:
            return {
                "exit_code": None,
                "stdout": _text(error.stdout),
                "stderr": _text(error.stderr)
                + f"\nExecution timed out after {RUN_TIMEOUT_SECONDS} seconds; the script process was stopped.",
                "timed_out": True,
            }
        except OSError:
            return {
                "exit_code": None, "stdout": "", "stderr": traceback.format_exc(),
                "timed_out": False,
            }

    return write_and_run

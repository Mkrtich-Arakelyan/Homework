# Silverbox homework

This repository contains the Silverbox neural ODE project in `task 5/` and the extended multi-agent system for Homework 2, Task 1 in `task 6/`.

## Task 1 system

`task 6/main.py` runs a LangGraph workflow with four agents: an orchestrator chooses the next step, a scientist supplies the scientific plan, a general agent explains supporting concepts, and an executor writes and runs Python scripts. A human node pauses the workflow for clarifications and approvals.

Execution and final reporting require explicit approval. Refusing an execution proposal records that the action was skipped. Each script has a 300-second timeout, and each run allows at most eight orchestrator decisions. Generated scripts run on CPU.

## Setup and running

Use Python 3.13, which was used for the recorded demonstrations, or a compatible Python interpreter. The following PowerShell commands are each a single line and assume the terminal is at the repository root.

Create a virtual environment if you do not already have one:

```powershell
py -3.13 -m venv .venv
```

Install the dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r '.\task 6\requirements.txt'
```

Create the local configuration only if it does not already exist:

```powershell
if (-not (Test-Path -LiteralPath '.\task 6\.env')) { Copy-Item -LiteralPath '.\task 6\.env.example' -Destination '.\task 6\.env' }
```

Set `OPENAI_API_KEY` in `task 6/.env`. The file stays local and is ignored by Git. The example configuration is shareable. The recorded demonstrations used `gpt-5-nano`; use `--model` to choose a different model.

Start an interactive run:

```powershell
.\.venv\Scripts\python.exe '.\task 6\main.py' --model gpt-5-nano
```

Enter your request at `Silverbox request:`. Answer clarification questions with ordinary text. At `Proceed? [yes/no, optional comment]:`, begin your reply with `yes` or `no`.

The default dataset is `task 6/SNLS80mV.mat`. To supply a different path:

```powershell
.\.venv\Scripts\python.exe '.\task 6\main.py' --data 'C:\path\to\SNLS80mV.mat' --model gpt-5-nano
```

## Recorded Task 1 evidence

| Role | Folder | Evidence |
| --- | --- | --- |
| Source model and recorded metrics | [task 6/runs/20261011_170944_164751](task%206/runs/20261011_170944_164751/) | Successful training script, checkpoint, metrics, and plot referenced by both demonstrations. |
| Part (a): ambiguous request | [task 6/runs/20261012_190611_631484](task%206/runs/20261012_190611_631484/) | The target RMSE was initially unspecified. The graph asked for it, the human supplied a target below 0.04 V, and the next orchestrator message used that target. This run finished. |
| Part (b): refusal followed by acceptance | [task 6/runs/20261012_191714_999190](task%206/runs/20261012_191714_999190/) | The human refused execution and requested an explanation. The graph recorded that the refused action did not run. A later execution approval was accepted, and the script read the saved metric and wrote `comparison_report.txt`. |

Each demonstration contains `dialogue.txt` for readable interaction evidence, `state.json` for the graph state and tool results, and `config.json` for the request and environment settings. Both demonstrations compare an already recorded metric; they do not perform a new model evaluation. The saved test RMSE is approximately 0.034322 V, below the supplied 0.04 V target.

**Part (b) limitation:** the approved script completed successfully, but printed `Saved value:`. The finish gate expects a test RMSE label, so the graph did not finish normally and stopped at its eight-decision limit. The original refusal, skipped-action message, later approval, and execution evidence are preserved. A correction and a new demonstration run are needed for a clean end-to-end completion.

The homework PDF must include the actual clarification question, human reply, and next dialogue message for part (a), and both execution approval questions, both replies, and the skipped-action message for part (b). The PDF has not yet been added.

## Git contents

`.gitignore` excludes local credentials, virtual environments, IDE settings, caches, temporary files, logs, root-level scratch runs, and new Task 6 run folders by default. It explicitly keeps the three evidence folders listed above available for commits. Task 5 datasets and evidence remain available.

When selecting a replacement demonstration, add its folder to the exceptions in `.gitignore`. Keep original transcripts intact, and select the required evidence for the PDF without rewriting the recorded dialogue.

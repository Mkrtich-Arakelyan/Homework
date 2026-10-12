import json
from pathlib import Path
from datetime import datetime

# 1) Paths
metrics_path = Path(r"C:\Users\User\Desktop\Homework\task 6\runs\20261011_170944_164751\metrics.json")
report_path = Path.cwd() / "comparison_report.txt"

# 2) Validate existence
if not metrics_path.is_file():
    raise FileNotFoundError(f"metrics.json not found at {metrics_path}")

# 3) Load JSON
with metrics_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

# 4) Extract RMSE with fallbacks
keys = ["retained_test_rmse_V", "retained_test_rmse", "test_rmse_V"]
value = None
for k in keys:
    if k in data:
        v = data[k]
        try:
            value = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"Value for {k} is not numeric: {v}")
        break

if value is None:
    raise KeyError("No retained_test_rmse_V (or legacy keys) found in metrics.json")

# 5) Compare to target
target = 0.04
result = "PASS" if value < target else "FAIL"

# 6) Timestamp
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 7) Write report (overwrite if exists)
report = (
    f"Saved retained_test_rmse_V: {value} V\n"
    f"Target RMSE: 0.04 V\n"
    f"Result: {result}\n"
    f"Timestamp: {timestamp}\n"
)
with report_path.open("w", encoding="utf-8") as f:
    f.write(report)

# 8) Print saved value
print(f"Saved value: {value} V")

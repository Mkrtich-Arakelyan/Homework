import json, os, math

# Configuration
path = r"C:\Users\User\Desktop\Homework\task 6\runs\20261011_170944_164751\metrics.json"
run_dir = r"C:\Users\User\Desktop\Homework\task 6\runs\20261011_170944_164751"
target_V = 0.04
anomalies = []

saved_value_V = None

# 1) Check file exists
if not os.path.exists(path):
    anomalies.append("file_not_found")
else:
    # 2) Load JSON
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        anomalies.append("json_parse_error: " + str(e))
        data = None
    except Exception as e:
        anomalies.append("read_error: " + str(e))
        data = None

    # 3) Locate retained_test_rmse_V
    def find_key(d, key):
        if isinstance(d, dict):
            if key in d:
                return d[key]
            for v in d.values():
                r = find_key(v, key)
                if r is not None:
                    return r
        elif isinstance(d, list):
            for item in d:
                r = find_key(item, key)
                if r is not None:
                    return r
        return None

    if data is not None:
        if isinstance(data, dict) and 'retained_test_rmse_V' in data:
            saved_value_V = data['retained_test_rmse_V']
        else:
            saved_value_V = find_key(data, 'retained_test_rmse_V')

    # 4) Validate numeric
    if saved_value_V is None:
        anomalies.append("missing_field_retained_test_rmse_V")
    else:
        if isinstance(saved_value_V, (int, float)):
            pass
        else:
            try:
                saved_value_V = float(saved_value_V)
                # keep as float
            except Exception:
                anomalies.append("non_numeric_retained_test_rmse_V")
                saved_value_V = None

# 5) Compute metrics if numeric
comparison_line = ""
summary = ""
meets_target = False
difference_V = None
if isinstance(saved_value_V, (int, float)):
    difference_V = target_V - float(saved_value_V)
    meets_target = float(saved_value_V) < target_V
    comparison_line = (
        f"Saved retained_test_rmse_V = {float(saved_value_V):.6f} V; "
        f"Target = 0.04 V; Meets target (strictly less): {'Yes' if meets_target else 'No'}; "
        f"Difference (Target - Saved) = {difference_V:.6f} V."
    )
    if meets_target:
        summary = f"The saved test RMSE is {float(saved_value_V):.6f} V, which is below the target of 0.04 V."
    else:
        summary = f"The saved test RMSE is {float(saved_value_V):.6f} V, which is above the target of 0.04 V by {(-difference_V):.6f} V."
else:
    comparison_line = "Saved value unavailable; cannot compare."
    summary = "Could not compute comparison due to missing numeric saved_value."

# 6) Write artifacts
results_path = os.path.join(run_dir, "results_metrics_check.txt")
with open(results_path, 'w') as rf:
    rf.write("Saved_value_V: {}
".format(saved_value_V if isinstance(saved_value_V, (int, float)) else "N/A"))
    rf.write("Target_V: 0.04\n")
    rf.write("Meets_target: {}\n".format("Yes" if (isinstance(saved_value_V, (int, float)) and saved_value_V < target_V) else "No"))
    rf.write("Difference_V: {}\n".format((f"{difference_V:.6f}" if isinstance(difference_V, (int, float)) else "N/A")))
    rf.write("Anomalies: {}\n".format("; ".join(anomalies) if anomalies else "None"))
    rf.write("Comparison_line: {}\n".format(comparison_line))
    rf.write("Summary: {}\n".format(summary))

# 7) Output
if isinstance(saved_value_V, (int, float)):
    print(f"Test RMSE: {float(saved_value_V):.6f} V")
else:
    print("Test RMSE: N/A V")
print(summary)

import json, os, sys
path = r"C:\Users\User\Desktop\Homework\task 6\runs\20261011_170944_164751\metrics.json"
run_dir = r"C:\Users\User\Desktop\Homework\task 6\runs\20261011_170944_164751"
target_V = 0.04
anomalies = []

saved_value_V = None
if not os.path.exists(path):
    anomalies.append("file_not_found")
else:
    data = None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        anomalies.append("json_parse_error: {}".format(e))
        data = None
    except Exception as e:
        anomalies.append("read_error: {}".format(e))
        data = None

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

    if saved_value_V is None:
        anomalies.append("missing_field_retained_test_rmse_V")
    else:
        if isinstance(saved_value_V, (int, float)):
            saved_value_V = float(saved_value_V)
        else:
            try:
                saved_value_V = float(saved_value_V)
            except Exception:
                anomalies.append("non_numeric_retained_test_rmse_V")
                saved_value_V = None

comparison_line = ""
summary = ""
meets_target = False
difference_V = None
if isinstance(saved_value_V, (int, float)):
    difference_V = target_V - saved_value_V
    meets_target = saved_value_V < target_V
    comparison_line = (f"Saved retained_test_rmse_V = {saved_value_V:.6f} V; Target = 0.04 V; Meets target (strictly less): {'Yes' if meets_target else 'No'}; Difference (Target - Saved) = {difference_V:.6f} V.")
    if meets_target:
        summary = f"The saved test RMSE is {saved_value_V:.6f} V, which is below the target of 0.04 V."
    else:
        summary = f"The saved test RMSE is {saved_value_V:.6f} V, which is above the target of 0.04 V by {(-difference_V):.6f} V."
else:
    comparison_line = "Saved value unavailable; cannot compare."
    summary = "Could not compute comparison due to missing numeric saved_value."

results_path = os.path.join(run_dir, "results_metrics_check.txt")
try:
    os.makedirs(run_dir, exist_ok=True)
except Exception:
    pass

with open(results_path, 'w') as rf:
    if isinstance(saved_value_V, (int, float)):
        rf.write("Saved_value_V: {:.6f}\n".format(saved_value_V))
    else:
        rf.write("Saved_value_V: N/A\n")
    rf.write("Target_V: 0.04\n")
    rf.write("Meets_target: {}\n".format("Yes" if isinstance(saved_value_V, (int, float)) and saved_value_V < target_V else "No"))
    rf.write("Difference_V: {}\n".format("{:.6f}".format(difference_V) if isinstance(difference_V, (int, float)) else "N/A"))
    rf.write("Anomalies: {}\n".format("; ".join(anomalies) if anomalies else "None"))
    rf.write("Comparison_line: {}\n".format(comparison_line))
    rf.write("Summary: {}\n".format(summary))

if isinstance(saved_value_V, (int, float)):
    print("Test RMSE: {:.6f} V".format(saved_value_V))
else:
    print("Test RMSE: N/A V")
print(summary)

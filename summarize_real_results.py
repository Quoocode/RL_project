import csv
import os
from collections import defaultdict

RESULTS_DIR = "results"

SERVICE_CSV = os.path.join(RESULTS_DIR, "real_deployment_services.csv")
COST_CSV = os.path.join(RESULTS_DIR, "real_deployment_weighted_cost.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "real_deployment_summary.csv")


def read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_summary():
    service_rows = read_csv(SERVICE_CSV)
    cost_rows = read_csv(COST_CSV)

    services_by_run = defaultdict(list)
    costs_by_run = defaultdict(list)

    for row in service_rows:
        services_by_run[row["run_id"]].append(row)

    for row in cost_rows:
        costs_by_run[row["run_id"]].append(row)

    summary_rows = []

    for run_id, rows in services_by_run.items():
        rows = sorted(rows, key=lambda r: int(r["service_index"]))

        mode = rows[0]["mode"]
        model_path = rows[0]["model_path"]

        total_services = len(rows)
        success_count = sum(1 for r in rows if r["success"] == "True")
        fallback_count = sum(1 for r in rows if r["fallback_used"] == "True")

        success_rate = success_count / total_services if total_services > 0 else 0.0

        placement_parts = []
        for r in rows:
            node = r["actual_node"] or r["final_node"]
            node_short = node.split("-")[-1] if node else "NA"
            placement_parts.append(f"{r['service_name']}->{node_short}")

        placement = "; ".join(placement_parts)

        run_cost_rows = costs_by_run.get(run_id, [])

        if run_cost_rows:
            total_weighted_cost = run_cost_rows[0]["total_cost"]
            valid_edges = run_cost_rows[0]["valid_edges"]
        else:
            total_weighted_cost = ""
            valid_edges = "0"

        summary_rows.append({
            "run_id": run_id,
            "mode": mode,
            "model_path": model_path,
            "total_services": total_services,
            "success_count": success_count,
            "success_rate": f"{success_rate:.2f}",
            "fallback_count": fallback_count,
            "valid_edges": valid_edges,
            "total_weighted_cost": total_weighted_cost,
            "placement": placement,
        })

    summary_rows = sorted(summary_rows, key=lambda r: r["run_id"])

    os.makedirs(RESULTS_DIR, exist_ok=True)

    fieldnames = [
        "run_id",
        "mode",
        "model_path",
        "total_services",
        "success_count",
        "success_rate",
        "fallback_count",
        "valid_edges",
        "total_weighted_cost",
        "placement",
    ]

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_rows


if __name__ == "__main__":
    summary = build_summary()

    print("\nREAL DEPLOYMENT SUMMARY")
    print("=" * 100)

    for row in summary:
        print(
            f"{row['run_id']} | "
            f"mode={row['mode']} | "
            f"success={row['success_count']}/{row['total_services']} | "
            f"fallback={row['fallback_count']} | "
            f"edges={row['valid_edges']} | "
            f"cost={row['total_weighted_cost']} | "
            f"{row['placement']}"
        )

    print("=" * 100)
    print(f"Đã lưu summary: {SUMMARY_CSV}")
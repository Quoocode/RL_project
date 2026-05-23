import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from datetime import datetime


DEPENDENCIES = [
    ("frontend", "auth", 0.90),
    ("frontend", "catalog", 0.80),
    ("catalog", "payment", 0.70),
    ("payment", "database", 0.95),
    ("auth", "database", 0.40),
]


def run_cmd(cmd, input_text=None, timeout=60):
    result = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout


def get_pods(namespace):
    raw = run_cmd(
        ["kubectl", "get", "pods", "-n", namespace, "-o", "json"],
        timeout=60,
    )
    data = json.loads(raw)

    pods = {}

    for item in data.get("items", []):
        name = item["metadata"]["name"]
        phase = item.get("status", {}).get("phase", "")
        pod_ip = item.get("status", {}).get("podIP", "")
        node_name = item.get("spec", {}).get("nodeName", "")

        ready = False
        for cond in item.get("status", {}).get("conditions", []):
            if cond.get("type") == "Ready" and cond.get("status") == "True":
                ready = True

        pods[name] = {
            "name": name,
            "phase": phase,
            "ready": ready,
            "ip": pod_ip,
            "node": node_name,
        }

    return pods


def ensure_probe(namespace, probe_name, node_name):
    manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": probe_name,
            "namespace": namespace,
            "labels": {
                "app": "latency-probe",
            },
        },
        "spec": {
            "nodeName": node_name,
            "restartPolicy": "Never",
            "containers": [
                {
                    "name": "curl",
                    "image": "curlimages/curl:8.10.1",
                    "command": ["sleep", "3600"],
                }
            ],
        },
    }

    run_cmd(
        ["kubectl", "apply", "-f", "-"],
        input_text=json.dumps(manifest),
        timeout=60,
    )

    run_cmd(
        [
            "kubectl", "wait",
            f"pod/{probe_name}",
            "-n", namespace,
            "--for=condition=Ready",
            "--timeout=120s",
        ],
        timeout=130,
    )


def delete_probe(namespace, probe_name):
    subprocess.run(
        [
            "kubectl", "delete", "pod", probe_name,
            "-n", namespace,
            "--ignore-not-found=true",
            "--wait=false",
        ],
        capture_output=True,
        text=True,
    )


def measure_edge(namespace, probe_name, dst_ip, samples):
    shell = (
        f"for i in $(seq 1 {samples}); do "
        f"curl -o /dev/null -s -w '%{{time_total}}\\n' "
        f"--connect-timeout 2 --max-time 5 http://{dst_ip}:80 "
        f"|| echo ERR; "
        f"done"
    )

    out = run_cmd(
        [
            "kubectl", "exec", "-n", namespace, probe_name,
            "--", "sh", "-c", shell,
        ],
        timeout=max(60, samples * 6),
    )

    latencies_ms = []
    errors = 0

    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue

        if line == "ERR":
            errors += 1
            continue

        try:
            # curl time_total is in seconds
            latencies_ms.append(float(line) * 1000.0)
        except ValueError:
            errors += 1

    return latencies_ms, errors


def percentile(values, p):
    if not values:
        return ""

    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)

    if f == c:
        return values[f]

    return values[f] + (values[c] - values[f]) * (k - f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="drl-scheduler")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", default="results/real_inter_service_latency.csv")
    parser.add_argument("--keep-probes", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    pods = get_pods(args.namespace)

    required = sorted(set([s for s, _, _ in DEPENDENCIES] + [d for _, d, _ in DEPENDENCIES]))

    missing = []
    for name in required:
        pod = pods.get(name)
        if not pod or pod["phase"] != "Running" or not pod["ready"] or not pod["ip"]:
            missing.append(name)

    if missing:
        raise RuntimeError(
            f"Missing or not Ready pods: {missing}. "
            f"Run live deployment first and check kubectl get pods -n {args.namespace} -o wide."
        )

    print("\nPODS")
    print("=" * 80)
    for name in required:
        p = pods[name]
        print(f"{name:<10} ip={p['ip']:<15} node={p['node']}")
    print("=" * 80)

    # Create one probe per source service, pinned to source service node
    source_services = sorted(set(src for src, _, _ in DEPENDENCIES))
    probe_by_src = {}

    try:
        for src in source_services:
            src_node = pods[src]["node"]
            probe_name = f"latency-probe-{src}"

            print(f"\nCreating probe {probe_name} on same node as {src}: {src_node}")
            ensure_probe(args.namespace, probe_name, src_node)
            probe_by_src[src] = probe_name

        rows = []

        print("\nMEASURING INTER-SERVICE LATENCY")
        print("=" * 100)

        for src, dst, weight in DEPENDENCIES:
            src_pod = pods[src]
            dst_pod = pods[dst]
            probe_name = probe_by_src[src]

            print(f"\n{src} -> {dst}")
            print(f"  probe      : {probe_name}")
            print(f"  src node   : {src_pod['node']}")
            print(f"  dst node   : {dst_pod['node']}")
            print(f"  dst ip     : {dst_pod['ip']}")
            print(f"  samples    : {args.samples}")

            latencies, errors = measure_edge(
                namespace=args.namespace,
                probe_name=probe_name,
                dst_ip=dst_pod["ip"],
                samples=args.samples,
            )

            success_count = len(latencies)

            if success_count > 0:
                avg_ms = statistics.mean(latencies)
                min_ms = min(latencies)
                max_ms = max(latencies)
                p50_ms = percentile(latencies, 50)
                p95_ms = percentile(latencies, 95)
                weighted_avg = weight * avg_ms
                weighted_p95 = weight * p95_ms
            else:
                avg_ms = min_ms = max_ms = p50_ms = p95_ms = ""
                weighted_avg = weighted_p95 = ""

            same_node = src_pod["node"] == dst_pod["node"]

            print(
                f"  success={success_count}/{args.samples}, errors={errors}, "
                f"avg={avg_ms:.3f} ms, p50={p50_ms:.3f} ms, p95={p95_ms:.3f} ms"
                if success_count > 0
                else f"  success=0/{args.samples}, errors={errors}"
            )

            rows.append({
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "namespace": args.namespace,
                "src": src,
                "dst": dst,
                "traffic_weight": weight,
                "src_pod_ip": src_pod["ip"],
                "dst_pod_ip": dst_pod["ip"],
                "src_node": src_pod["node"],
                "dst_node": dst_pod["node"],
                "same_node": same_node,
                "probe_pod": probe_name,
                "samples": args.samples,
                "success_count": success_count,
                "error_count": errors,
                "avg_latency_ms": avg_ms,
                "p50_latency_ms": p50_ms,
                "p95_latency_ms": p95_ms,
                "min_latency_ms": min_ms,
                "max_latency_ms": max_ms,
                "weighted_avg_latency": weighted_avg,
                "weighted_p95_latency": weighted_p95,
            })

        fieldnames = [
            "run_id",
            "timestamp",
            "namespace",
            "src",
            "dst",
            "traffic_weight",
            "src_pod_ip",
            "dst_pod_ip",
            "src_node",
            "dst_node",
            "same_node",
            "probe_pod",
            "samples",
            "success_count",
            "error_count",
            "avg_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "min_latency_ms",
            "max_latency_ms",
            "weighted_avg_latency",
            "weighted_p95_latency",
        ]

        file_exists = os.path.exists(args.output)

        with open(args.output, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerows(rows)

        total_weighted_avg = sum(
            float(r["weighted_avg_latency"])
            for r in rows
            if r["weighted_avg_latency"] != ""
        )

        total_weighted_p95 = sum(
            float(r["weighted_p95_latency"])
            for r in rows
            if r["weighted_p95_latency"] != ""
        )

        print("\nSUMMARY")
        print("=" * 80)
        print(f"run_id              : {run_id}")
        print(f"output              : {args.output}")
        print(f"total_weighted_avg  : {total_weighted_avg:.3f} ms")
        print(f"total_weighted_p95  : {total_weighted_p95:.3f} ms")
        print("=" * 80)

    finally:
        if not args.keep_probes:
            print("\nCleaning probe pods...")
            for probe_name in probe_by_src.values():
                delete_probe(args.namespace, probe_name)


if __name__ == "__main__":
    main()
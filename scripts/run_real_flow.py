# scripts/run_real_flow.py
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_NAMESPACE = "drl-scheduler"
DEFAULT_CLUSTER_STATE = "cluster_state.json"
DEFAULT_LATENCY_YAML = "latency-test.yaml"
DEFAULT_LATENCY_JSON = "real_node_latency_matrix.json"
DEFAULT_LATENCY_CSV = "real_node_latency_matrix.csv"


def run_cmd(cmd, capture=True, check=True, show_output=True):
    """
    Run shell command.

    show_output=False dùng cho các lệnh có output rất dài như:
    - kubectl get pods -o json
    - ping
    """
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=capture,
    )

    if show_output and capture:
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )

    return result.stdout if capture else ""


def kubectl(*args, capture=True, check=True, show_output=True):
    return run_cmd(
        ["kubectl", *args],
        capture=capture,
        check=check,
        show_output=show_output,
    )


def ensure_namespace(namespace: str):
    kubectl("create", "namespace", namespace, check=False, show_output=False)


def get_nodes_by_label(size: str):
    out = kubectl(
        "get",
        "nodes",
        "-l",
        f"node-size={size}",
        "-o",
        r"jsonpath={range .items[*]}{.metadata.name}{'\n'}{end}",
        show_output=False,
    )

    nodes = [line.strip() for line in out.splitlines() if line.strip()]
    return sorted(nodes)


def short_node_name(name: str) -> str:
    """
    Rút gọn node name để log dễ đọc.
    """
    if not name:
        return name

    parts = name.split("-")
    if len(parts) <= 3:
        return name

    # Ví dụ: gke-rl-placement-cluster-large-pool-ba1c40dc-5zc6
    # -> ...large-pool-5zc6
    if "large" in name:
        return f"...large-pool-{parts[-1]}"
    if "medium" in name:
        return f"...medium-pool-{parts[-1]}"
    if "default" in name:
        return f"...small-pool-{parts[-1]}"

    return f"...{parts[-1]}"


def export_cluster_state(namespace: str, cluster_state_path: str):
    """
    Đọc node đã có trên cluster hiện tại.

    Yêu cầu cluster đã được tạo trước đó với labels:
      node-size=small
      node-size=medium
      node-size=large

    Node order khớp topology train:
      Node 0: small
      Node 1: medium
      Node 2: large
      Node 3: medium
      Node 4: small
    """

    print("\n=== Node order ===")

    small_nodes = get_nodes_by_label("small")
    medium_nodes = get_nodes_by_label("medium")
    large_nodes = get_nodes_by_label("large")

    if len(small_nodes) < 2:
        raise RuntimeError(f"Need at least 2 small nodes, found {len(small_nodes)}")
    if len(medium_nodes) < 2:
        raise RuntimeError(f"Need at least 2 medium nodes, found {len(medium_nodes)}")
    if len(large_nodes) < 1:
        raise RuntimeError(f"Need at least 1 large node, found {len(large_nodes)}")

    node_order = [
        {"index": 0, "role": "small", "name": small_nodes[0]},
        {"index": 1, "role": "medium", "name": medium_nodes[0]},
        {"index": 2, "role": "large", "name": large_nodes[0]},
        {"index": 3, "role": "medium", "name": medium_nodes[1]},
        {"index": 4, "role": "small", "name": small_nodes[1]},
    ]

    state = {
        "namespace": namespace,
        "node_order": node_order,
    }

    Path(cluster_state_path).write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for node in node_order:
        print(
            f"Node {node['index']} "
            f"{node['role']:<6} -> {short_node_name(node['name'])}"
        )

    print(f"Saved: {cluster_state_path}")

    return state


def generate_latency_yaml(cluster_state: dict, output_yaml_path: str):
    namespace = cluster_state["namespace"]
    docs = []

    for node in sorted(cluster_state["node_order"], key=lambda x: int(x["index"])):
        idx = int(node["index"])
        role = node["role"]
        node_name = node["name"]

        doc = f"""apiVersion: v1
kind: Pod
metadata:
  name: netshoot-node{idx}
  namespace: {namespace}
  labels:
    app: latency-test
    node-index: "{idx}"
    node-role: "{role}"
spec:
  nodeName: {node_name}
  restartPolicy: Always
  containers:
  - name: netshoot
    image: nicolaka/netshoot
    command: ["sleep", "36000"]
"""
        docs.append(doc)

    Path(output_yaml_path).write_text("---\n".join(docs), encoding="utf-8")


def cleanup_latency_pods(namespace: str):
    kubectl(
        "delete",
        "pod",
        "-n",
        namespace,
        "-l",
        "app=latency-test",
        "--ignore-not-found=true",
        check=False,
        show_output=False,
    )


def apply_latency_pods(latency_yaml_path: str):
    print("\n=== Prepare latency test pods ===")
    kubectl("apply", "-f", latency_yaml_path, show_output=False)


def wait_for_latency_pods(namespace: str, expected_count: int = 5, timeout_seconds: int = 180):
    print("\n=== Latency pods ===")

    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        out = kubectl(
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            "app=latency-test",
            "-o",
            "json",
            show_output=False,
        )

        data = json.loads(out)
        items = data.get("items", [])

        running = [
            item
            for item in items
            if item.get("status", {}).get("phase") == "Running"
        ]

        print(f"Running: {len(running)}/{expected_count}")

        if len(running) >= expected_count:
            print_latency_pod_table(namespace)
            return

        time.sleep(5)

    raise TimeoutError("Timeout waiting for latency-test pods to become Running")


def print_latency_pod_table(namespace: str):
    out = kubectl(
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        "app=latency-test",
        "-o",
        "custom-columns=NAME:.metadata.name,STATUS:.status.phase,IP:.status.podIP,NODE:.spec.nodeName",
        show_output=False,
    )

    lines = [line for line in out.splitlines() if line.strip()]
    if not lines:
        return

    print("Latency test pods:")
    print(lines[0])

    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 4:
            name = parts[0]
            status = parts[1]
            ip = parts[2]
            node = parts[3]
            print(f"{name:<16} {status:<8} {ip:<12} {short_node_name(node)}")
        else:
            print(line)


def get_pod_ip(namespace: str, pod_name: str):
    ip = kubectl(
        "get",
        "pod",
        pod_name,
        "-n",
        namespace,
        "-o",
        r"jsonpath={.status.podIP}",
        show_output=False,
    ).strip()

    if not ip:
        raise RuntimeError(f"Pod {pod_name} has no IP")

    return ip


def parse_ping_avg_ms(output: str):
    match = re.search(r"(?:rtt|round-trip).*=\s*([\d.]+)/([\d.]+)/([\d.]+)/", output)
    if not match:
        return None
    return float(match.group(2))


def measure_latency_matrix(
    cluster_state: dict,
    ping_count: int,
    output_json: str,
    output_csv: str,
):
    print("\n=== Measure latency matrix ===")

    namespace = cluster_state["namespace"]
    node_order = sorted(cluster_state["node_order"], key=lambda x: int(x["index"]))

    pods = []

    for node in node_order:
        idx = int(node["index"])
        pod_name = f"netshoot-node{idx}"
        ip = get_pod_ip(namespace, pod_name)

        pods.append(
            {
                "index": idx,
                "pod": pod_name,
                "ip": ip,
                "node": node["name"],
                "role": node["role"],
            }
        )

    n = len(pods)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    rows = [["src", "dst", "avg_ms"]]

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0.0
                rows.append([i, j, "0.000000"])
                continue

            src_pod = pods[i]["pod"]
            dst_ip = pods[j]["ip"]

            out = kubectl(
                "exec",
                "-n",
                namespace,
                src_pod,
                "--",
                "ping",
                "-c",
                str(ping_count),
                dst_ip,
                check=False,
                show_output=False,
            )

            avg = parse_ping_avg_ms(out)

            if avg is None:
                avg = 1.0
                print(f"  {i}->{j}: parse failed, fallback {avg:.3f} ms")
            else:
                print(f"  {i}->{j}: {avg:.3f} ms")

            matrix[i][j] = avg
            rows.append([i, j, f"{avg:.6f}"])

    result = {
        "namespace": namespace,
        "ping_count": ping_count,
        "node_order": node_order,
        "pods": pods,
        "latency_ms": matrix,
    }

    Path(output_json).write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("\nLatency matrix ms:")
    for row in matrix:
        print("  " + "  ".join(f"{x:.3f}" for x in row))

    print(f"Saved: {output_json}")
    print(f"Saved: {output_csv}")


def cleanup_old_service_pods(namespace: str):
    print("\n=== Cleanup old DQN service pods ===")

    out = kubectl(
        "delete",
        "pod",
        "-n",
        namespace,
        "-l",
        "app=drl-placed",
        "--ignore-not-found=true",
        check=False,
        show_output=False,
    )

    if out.strip():
        print(out.strip())
    else:
        print("No old DQN service pods")


def run_scheduler(args):
    print("\n=== Run DQN scheduler ===")

    cmd = [
        sys.executable,
        "..\\k8s_real_env.py",
    ]

    if args.dry_run:
        cmd.append("--dry-run")

    cmd.extend(
        [
            "--services",
            str(args.services),
            "--model",
            args.model,
            "--resource-scale",
            str(args.resource_scale),
            "--cluster-state",
            args.cluster_state,
            "--latency-matrix",
            args.latency_json,
        ]
    )

    if args.debug_print_obs:
        cmd.append("--debug-print-obs")

    run_cmd(cmd, capture=False)


def print_final_hint(args):
    print("\n=== Done ===")
    print("Useful next commands:")
    print(f"  kubectl get pods -n {args.namespace} -o wide")
    print(f"  python .\\cleanup_real_flow.py --services-only")
    print(f"  python .\\cleanup_real_flow.py")


def main():
    parser = argparse.ArgumentParser(
        description="Run real GKE experiment flow without creating/deleting cluster"
    )

    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--cluster-state", default=DEFAULT_CLUSTER_STATE)
    parser.add_argument("--latency-yaml", default=DEFAULT_LATENCY_YAML)
    parser.add_argument("--latency-json", default=DEFAULT_LATENCY_JSON)
    parser.add_argument("--latency-csv", default=DEFAULT_LATENCY_CSV)

    parser.add_argument("--model", default="..\\models\\dqn_model_seed789")
    parser.add_argument("--services", type=int, default=5)
    parser.add_argument("--resource-scale", type=float, default=1.0)
    parser.add_argument("--ping-count", type=int, default=10)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-print-obs", action="store_true")

    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Reuse existing real_node_latency_matrix.json instead of measuring again",
    )

    parser.add_argument(
        "--keep-latency-pods",
        action="store_true",
        help="Do not recreate latency-test pods",
    )

    args = parser.parse_args()

    ensure_namespace(args.namespace)

    cluster_state = export_cluster_state(
        namespace=args.namespace,
        cluster_state_path=args.cluster_state,
    )

    if not args.skip_latency:
        if not args.keep_latency_pods:
            cleanup_latency_pods(args.namespace)
            generate_latency_yaml(cluster_state, args.latency_yaml)
            apply_latency_pods(args.latency_yaml)
            wait_for_latency_pods(
                args.namespace,
                expected_count=len(cluster_state["node_order"]),
            )
        else:
            print("\n=== Keep existing latency-test pods ===")
            wait_for_latency_pods(
                args.namespace,
                expected_count=len(cluster_state["node_order"]),
            )

        measure_latency_matrix(
            cluster_state=cluster_state,
            ping_count=args.ping_count,
            output_json=args.latency_json,
            output_csv=args.latency_csv,
        )
    else:
        print("\n=== Reuse existing latency matrix ===")
        if not Path(args.latency_json).exists():
            raise FileNotFoundError(
                f"{args.latency_json} not found but --skip-latency was provided"
            )
        print(f"Using: {args.latency_json}")

    cleanup_old_service_pods(args.namespace)
    run_scheduler(args)
    print_final_hint(args)


if __name__ == "__main__":
    main()
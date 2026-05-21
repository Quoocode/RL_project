import json
from pathlib import Path


CLUSTER_STATE_PATH = Path("cluster_state.json")
OUTPUT_YAML_PATH = Path("latency-test.yaml")


def main():
    if not CLUSTER_STATE_PATH.exists():
        raise FileNotFoundError(
            "cluster_state.json not found. Run provision_gke.ps1 first."
        )

    with CLUSTER_STATE_PATH.open("r", encoding="utf-8") as f:
        state = json.load(f)

    namespace = state.get("namespace", "drl-scheduler")
    node_order = sorted(state["node_order"], key=lambda x: int(x["index"]))

    docs = []

    for node in node_order:
        idx = int(node["index"])
        node_name = node["name"]
        role = node.get("role", "unknown")

        pod_yaml = f"""apiVersion: v1
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
        docs.append(pod_yaml)

    OUTPUT_YAML_PATH.write_text("---\n".join(docs), encoding="utf-8")

    print(f"Saved {OUTPUT_YAML_PATH}")
    print(f"Namespace: {namespace}")
    print("Node order:")
    for node in node_order:
        print(f"  Node {node['index']} ({node['role']}) -> {node['name']}")


if __name__ == "__main__":
    main()
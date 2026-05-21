# k8s_real_env.py
"""
Bridge layer: kết nối DRL agent đã train với Kubernetes cluster thật.

Kiến trúc:
  Real K8s Cluster
       │  kubectl top nodes
       ▼
  RealClusterObserver  → obs vector 17-dim (khớp model đã train)
       │  model.predict(obs)
       ▼
  DRL Agent (DQN)      → node_id (0-4)
       │  map → node name thật
       ▼
  KubectlExecutor      → kubectl apply pod với nodeName
       │
       ▼
  Real K8s Cluster     ← pod chạy đúng node

Chạy dry-run (không deploy thật):
    python k8s_real_env.py --dry-run

Chạy thật:
    python k8s_real_env.py

Chạy nhiều services:
    python k8s_real_env.py --services 5 --model ./models/dqn_model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import subprocess
import time
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Optional


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG — khớp với cluster thật
# ═════════════════════════════════════════════════════════════════════════════

# Khớp với model DQN env v3 đã train cho kịch bản 5 nodes / 5 services
NUM_NODES     = 5
NUM_SERVICES  = 5

# Env v3 observation:
# obs_dim = NUM_NODES*4 + 2 + NUM_SERVICES = 27
EXPECTED_OBS_DIM = (NUM_NODES * 4) + 2 + NUM_SERVICES

# Fallback capacity, chỉ dùng khi không đọc được allocatable.
# Không dùng các số này làm nguồn chính.
DEFAULT_NODE_CPU_CAPACITY_CORES = 1.0
DEFAULT_NODE_MEM_CAPACITY_MB    = 1024.0

# Namespace riêng để không ảnh hưởng cluster
K8S_NAMESPACE = "drl-scheduler"


# ═════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeMetrics:
    name: str
    cpu_used_cores: float
    mem_used_mb: float
    cpu_capacity: float = DEFAULT_NODE_CPU_CAPACITY_CORES
    mem_capacity: float = DEFAULT_NODE_MEM_CAPACITY_MB

    @property
    def cpu_util(self) -> float:
        return min(self.cpu_used_cores / self.cpu_capacity, 1.0)

    @property
    def mem_util(self) -> float:
        return min(self.mem_used_mb / self.mem_capacity, 1.0)


@dataclass
class ServiceRequest:
    name: str
    cpu_cores: float   # vd: 0.5 = 500m
    mem_mb: float      # vd: 512
    image: str = "nginx:alpine"


# ═════════════════════════════════════════════════════════════════════════════
# CLUSTER OBSERVER
# ═════════════════════════════════════════════════════════════════════════════

class RealClusterObserver:
    """
    Đọc trạng thái cluster thật.

    Bản mới:
    - Không hard-code node name.
    - Tự discover đúng NUM_NODES node từ cluster hiện tại.
    - Tự đọc allocatable CPU/RAM từ kubectl get nodes -o json.
    - Build observation 27-dim khớp env v3.
    """

    def __init__(self, expected_nodes: int = NUM_NODES):
        self.expected_nodes = expected_nodes
        self.worker_nodes = self._discover_worker_nodes()
        self.capacity_map = self._discover_node_capacities()

        print("\n  ✅ Discovered worker nodes:")
        for i, node in enumerate(self.worker_nodes):
            cap = self.capacity_map[node]
            print(
                f"    [{i}] {node} | "
                f"CPU={cap['cpu']} cores | MEM={cap['mem']} MB"
            )

    def _run_kubectl_json(self, args: List[str]) -> dict:
        result = subprocess.run(
            ["kubectl"] + args,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

        return json.loads(result.stdout)

    def _parse_cpu_to_cores(self, cpu_str: str) -> float:
        """
        Kubernetes CPU:
        - 940m -> 0.94
        - 2    -> 2.0
        """
        if cpu_str.endswith("m"):
            return float(cpu_str[:-1]) / 1000.0
        return float(cpu_str)

    def _parse_mem_to_mb(self, mem_str: str) -> float:
        """
        Kubernetes memory:
        - Ki -> MB
        - Mi -> MB
        - Gi -> MB
        """
        if mem_str.endswith("Ki"):
            return float(mem_str[:-2]) / 1024.0
        if mem_str.endswith("Mi"):
            return float(mem_str[:-2])
        if mem_str.endswith("Gi"):
            return float(mem_str[:-2]) * 1024.0

        # fallback: assume bytes
        return float(mem_str) / (1024.0 * 1024.0)

    def _is_ready_node(self, node: dict) -> bool:
        conditions = node.get("status", {}).get("conditions", [])
        for cond in conditions:
            if cond.get("type") == "Ready":
                return cond.get("status") == "True"
        return False

    def _discover_worker_nodes(self) -> List[str]:
        """
        Tự động lấy danh sách node từ cluster hiện tại.

        Với GKE managed cluster, các node trả về thường là worker nodes.
        Ta lấy các node Ready và sort theo tên để mapping action ổn định:
          action 0 -> node[0]
          action 1 -> node[1]
          ...
        """
        data = self._run_kubectl_json(["get", "nodes", "-o", "json"])

        nodes = []
        for item in data.get("items", []):
            name = item["metadata"]["name"]

            # Chỉ lấy node Ready
            if not self._is_ready_node(item):
                continue

            nodes.append(name)

        nodes = sorted(nodes)

        if len(nodes) != self.expected_nodes:
            raise RuntimeError(
                f"Expected {self.expected_nodes} Ready nodes for DQN "
                f"5n5 model, but found {len(nodes)}: {nodes}. "
                f"Create a {self.expected_nodes}-node cluster or use the "
                f"matching model."
            )

        return nodes

    def _discover_node_capacities(self) -> dict:
        """
        Đọc allocatable CPU/RAM của từng node.

        Dùng allocatable thay vì capacity vì Kubernetes scheduler cũng dựa
        trên tài nguyên có thể cấp phát thực tế.
        """
        data = self._run_kubectl_json(["get", "nodes", "-o", "json"])

        capacity_map = {}

        for item in data.get("items", []):
            name = item["metadata"]["name"]

            if name not in self.worker_nodes:
                continue

            alloc = item.get("status", {}).get("allocatable", {})

            cpu = self._parse_cpu_to_cores(
                alloc.get("cpu", str(DEFAULT_NODE_CPU_CAPACITY_CORES))
            )

            mem = self._parse_mem_to_mb(
                alloc.get("memory", f"{int(DEFAULT_NODE_MEM_CAPACITY_MB)}Mi")
            )

            capacity_map[name] = {
                "cpu": cpu,
                "mem": mem,
            }

        for node in self.worker_nodes:
            if node not in capacity_map:
                capacity_map[node] = {
                    "cpu": DEFAULT_NODE_CPU_CAPACITY_CORES,
                    "mem": DEFAULT_NODE_MEM_CAPACITY_MB,
                }

        return capacity_map

    def get_metrics(self) -> List[NodeMetrics]:
        """
        Parse output của kubectl top nodes.
        CPU "27m" = 27 millicores = 0.027 cores.
        MEM "177Mi" = 177 MB.
        """
        try:
            result = subprocess.run(
                ["kubectl", "top", "nodes", "--no-headers"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                raise RuntimeError(result.stderr)

            metrics_map = {}

            for line in result.stdout.strip().split("\n"):
                parts = line.split()
                if len(parts) < 4:
                    continue

                name = parts[0]

                if name not in self.worker_nodes:
                    continue

                cpu_str = parts[1]
                mem_str = parts[3]

                cpu_cores = self._parse_cpu_to_cores(cpu_str)
                mem_mb = self._parse_mem_to_mb(mem_str)

                cap = self.capacity_map[name]

                metrics_map[name] = NodeMetrics(
                    name=name,
                    cpu_used_cores=cpu_cores,
                    mem_used_mb=mem_mb,
                    cpu_capacity=cap["cpu"],
                    mem_capacity=cap["mem"],
                )

            metrics = []
            for node in self.worker_nodes:
                cap = self.capacity_map[node]
                metrics.append(
                    metrics_map.get(
                        node,
                        NodeMetrics(
                            name=node,
                            cpu_used_cores=0.0,
                            mem_used_mb=0.0,
                            cpu_capacity=cap["cpu"],
                            mem_capacity=cap["mem"],
                        ),
                    )
                )

            return metrics

        except Exception as e:
            print(f"  ⚠️  Không đọc được metrics ({e}), dùng 0%")

            metrics = []
            for node in self.worker_nodes:
                cap = self.capacity_map[node]
                metrics.append(
                    NodeMetrics(
                        name=node,
                        cpu_used_cores=0.0,
                        mem_used_mb=0.0,
                        cpu_capacity=cap["cpu"],
                        mem_capacity=cap["mem"],
                    )
                )
            return metrics

    def build_observation(self, metrics: List[NodeMetrics],
                          service: ServiceRequest,
                          service_idx: int,
                          session_cpu: dict = None,
                          session_mem: dict = None) -> np.ndarray:
        """
        Chuyển metrics thật → obs vector 27-dim cho model env v3.

        Format:
        [
          cpu_util_0, mem_util_0, cpu_capacity_norm_0, mem_capacity_norm_0,
          ...
          cpu_util_4, mem_util_4, cpu_capacity_norm_4, mem_capacity_norm_4,
          service_cpu_norm, service_mem_norm,
          onehot_service_0, ..., onehot_service_4
        ]
        """
        obs = []
        session_cpu = session_cpu or {}
        session_mem = session_mem or {}

        max_cpu_capacity = max(m.cpu_capacity for m in metrics)
        max_mem_capacity = max(m.mem_capacity for m in metrics)

        for m in metrics:
            extra_cpu = session_cpu.get(m.name, 0.0)
            extra_mem = session_mem.get(m.name, 0.0)

            adj_cpu = min(
                (m.cpu_used_cores + extra_cpu) / m.cpu_capacity,
                1.0,
            )
            adj_mem = min(
                (m.mem_used_mb + extra_mem) / m.mem_capacity,
                1.0,
            )

            cpu_capacity_norm = min(m.cpu_capacity / max_cpu_capacity, 1.0)
            mem_capacity_norm = min(m.mem_capacity / max_mem_capacity, 1.0)

            obs.extend([
                float(adj_cpu),
                float(adj_mem),
                float(cpu_capacity_norm),
                float(mem_capacity_norm),
            ])

        # Service resource request — normalize giống simulation env.
        # simulation dùng CPU cores / 2.0 và memory GB / 4.0.
        obs.append(min(service.cpu_cores / 2.0, 1.0))
        obs.append(min((service.mem_mb / 1024.0) / 4.0, 1.0))

        onehot = [0.0] * NUM_SERVICES
        if service_idx < NUM_SERVICES:
            onehot[service_idx] = 1.0
        obs.extend(onehot)

        obs = np.array(obs, dtype=np.float32)

        if obs.shape[0] != EXPECTED_OBS_DIM:
            raise ValueError(
                f"Observation shape mismatch: got {obs.shape[0]}, "
                f"expected {EXPECTED_OBS_DIM}"
            )

        return obs

    def print_cluster_state(self, metrics: List[NodeMetrics],
                             session_cpu: dict = None,
                             session_mem: dict = None):
        session_cpu = session_cpu or {}
        session_mem = session_mem or {}

        print(
            f"  {'Node':<55} {'CPU(adj)':>10} {'RAM(adj)':>10} "
            f"{'CPU cap':>8} {'RAM cap':>8}"
        )
        print(f"  {'-'*100}")

        for m in metrics:
            extra_cpu = session_cpu.get(m.name, 0.0)
            extra_mem = session_mem.get(m.name, 0.0)

            adj_cpu = min(
                (m.cpu_used_cores + extra_cpu) / m.cpu_capacity,
                1.0,
            )
            adj_mem = min(
                (m.mem_used_mb + extra_mem) / m.mem_capacity,
                1.0,
            )

            print(
                f"  {m.name:<55} "
                f"{adj_cpu*100:>8.1f}% "
                f"{adj_mem*100:>8.1f}% "
                f"{m.cpu_capacity:>8.2f} "
                f"{m.mem_capacity:>8.0f}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# KUBECTL EXECUTOR
# ═════════════════════════════════════════════════════════════════════════════

class KubectlExecutor:
    """Deploy pod lên đúng node qua kubectl."""

    def __init__(self, namespace: str = K8S_NAMESPACE, dry_run: bool = False):
        self.namespace = namespace
        self.dry_run   = dry_run
        if not dry_run:
            self._ensure_namespace()

    def _ensure_namespace(self):
        subprocess.run(
            ["kubectl", "create", "namespace", self.namespace],
            capture_output=True
        )

    def _make_pod_yaml(self, service: ServiceRequest,
                       node_name: str) -> str:
        """
        Tạo pod YAML với nodeName — đây là cách chỉ định
        chính xác node nào sẽ chạy pod này trong K8s.
        """
        cpu_req = f"{int(service.cpu_cores * 1000)}m"  # "500m"
        mem_req = f"{int(service.mem_mb)}Mi"            # "512Mi"

        return f"""apiVersion: v1
kind: Pod
metadata:
  name: {service.name}
  namespace: {self.namespace}
  labels:
    app: drl-placed
    service-name: {service.name}
spec:
  nodeName: {node_name}
  restartPolicy: Never
  containers:
  - name: {service.name}
    image: {service.image}
    resources:
      requests:
        cpu: "{cpu_req}"
        memory: "{mem_req}"
      limits:
        cpu: "{cpu_req}"
        memory: "{mem_req}"
"""

    def deploy(self, service: ServiceRequest, node_name: str) -> bool:
        if self.dry_run:
            print(f"    [DRY RUN] {service.name} → {node_name} "
                  f"({service.cpu_cores} CPU / {service.mem_mb} MB)")
            return True

        yaml_str = self._make_pod_yaml(service, node_name)
        try:
            result = subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=yaml_str, text=True,
                capture_output=True, timeout=15
            )
            return result.returncode == 0
        except Exception as e:
            print(f"    ❌ Deploy error: {e}")
            return False

    def delete_all_pods(self):
        """Xóa tất cả pods trong namespace (cleanup)."""
        if self.dry_run:
            print(f"  [DRY RUN] Delete all pods in {self.namespace}")
            return
        subprocess.run(
            ["kubectl", "delete", "pods", "--all",
             "-n", self.namespace, "--wait=false"],
            capture_output=True
        )
        print(f"  🗑️  Đã xóa tất cả pods trong namespace '{self.namespace}'")

    def get_pod_node(self, pod_name: str) -> str:
        """Verify pod đang chạy trên node nào."""
        result = subprocess.run(
            ["kubectl", "get", "pod", pod_name,
             "-n", self.namespace,
             "-o", "jsonpath={.spec.nodeName}"],
            capture_output=True, text=True
        )
        return result.stdout.strip()


# ═════════════════════════════════════════════════════════════════════════════
# REAL K8S SCHEDULER
# ═════════════════════════════════════════════════════════════════════════════

class RealK8sScheduler:
    """
    Main scheduler: dùng DRL agent để quyết định đặt service lên node nào,
    sau đó thực thi quyết định trên cluster thật.
    """

    def __init__(self, model_path: str, dry_run: bool = False):
        self.dry_run  = dry_run
        self.observer = RealClusterObserver()
        self.worker_nodes = self.observer.worker_nodes
        self.executor = KubectlExecutor(dry_run=dry_run)

        # Load model — import ở đây để không lỗi nếu SB3 chưa cài
        from stable_baselines3 import DQN
        print(f"\n  Đang load model: {model_path} ...")
        self.model = DQN.load(model_path)
        print(f"  ✅ Model loaded")

    def _node_id_to_name(self, node_id: int,
                     metrics: List[NodeMetrics]) -> str:
        """
        Map action index 0..4 sang node hiện tại đã discover.
        """
        if 0 <= node_id < len(self.worker_nodes):
            return self.worker_nodes[node_id]

        print(f"  ⚠️  node_id={node_id} out of range, fallback least-loaded")

        return min(
            self.worker_nodes,
            key=lambda n: next(
                (m.cpu_util for m in metrics if m.name == n),
                1.0,
            ),
        )

    def schedule(self, services: List[ServiceRequest]):
        """Chạy scheduling cho toàn bộ service chain."""

        print(f"\n{'═'*55}")
        print(f"  DRL K8s SCHEDULER")
        print(f"  {len(services)} services | "
              f"{'DRY RUN' if self.dry_run else 'LIVE'}")
        print(f"{'═'*55}")

        results = []

        # Track resource đã deploy trong session này
        # (bù cho metrics-server update chậm 15-60s)
        session_cpu: dict = {n: 0.0 for n in self.worker_nodes}
        session_mem: dict = {n: 0.0 for n in self.worker_nodes}

        for idx, service in enumerate(services):
            print(f"\n  ── Service {idx+1}/{len(services)}: "
                  f"{service.name} ──")
            print(f"     Yêu cầu: {service.cpu_cores} CPU cores | "
                  f"{service.mem_mb} MB RAM")

            # 1. Đọc trạng thái cluster thật
            metrics = self.observer.get_metrics()
            self.observer.print_cluster_state(
                metrics, session_cpu, session_mem
            )

            # 2. Tạo obs vector — truyền session tracking vào
            obs = self.observer.build_observation(
                metrics, service, idx, session_cpu, session_mem
            )

            expected_shape = self.model.observation_space.shape[0]
            if obs.shape[0] != expected_shape:
                raise ValueError(
                    f"Observation shape mismatch: real obs={obs.shape[0]}, "
                    f"model expects={expected_shape}. "
                    f"Check NUM_NODES/NUM_SERVICES and build_observation()."
                )

            # 3. Agent quyết định
            action, _ = self.model.predict(obs, deterministic=True)
            node_id   = int(action)
            node_name = self._node_id_to_name(node_id, metrics)

            print(f"\n     🤖 Agent chọn: Node {node_id} ({node_name})")

            # 4. Deploy
            success = self.executor.deploy(service, node_name)
            print(f"     {'✅ Deployed' if success else '❌ Failed'}")

            # 5. Cập nhật session tracking ngay sau deploy
            #    Không chờ metrics-server — tự cộng vào để bước sau thấy
            if success:
                session_cpu[node_name] += service.cpu_cores
                session_mem[node_name] += service.mem_mb

            results.append({
                "service"  : service.name,
                "node_id"  : node_id,
                "node_name": node_name,
                "success"  : success,
            })

            # Chờ ngắn để K8s xử lý pod
            if idx < len(services) - 1 and not self.dry_run:
                time.sleep(2)

        self._print_summary(results)
        return results

    def _print_summary(self, results: list):
        n_ok = sum(1 for r in results if r["success"])

        print(f"\n{'═'*55}")
        print(f"  KẾT QUẢ: {n_ok}/{len(results)} services scheduled")
        print(f"{'═'*55}")
        print(f"  {'Service':<20} {'Node':<15} Status")
        print(f"  {'-'*48}")
        for r in results:
            icon = "✅" if r["success"] else "❌"
            print(f"  {r['service']:<20} {r['node_name']:<15} {icon}")
        print(f"{'═'*55}")

        if n_ok > 0 and not self.dry_run:
            print(f"\n  Xem pods đã deploy:")
            print(f"  kubectl get pods -n {K8S_NAMESPACE} -o wide")
            print(f"\n  Xóa sau khi xem xong:")
            print(f"  kubectl delete pods --all -n {K8S_NAMESPACE}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real K8s Scheduler dùng DRL agent"
    )
    parser.add_argument("--model",    type=str,
                        default="./models/dqn_model",
                        help="Path đến model (không cần .zip)")
    parser.add_argument("--services", type=int, default=5,
                        help="Số services trong chain (default: 5)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--dry-run",  action="store_true",
                        help="Chạy thử, không deploy thật")
    parser.add_argument("--cleanup",  action="store_true",
                        help="Xóa tất cả pods trong namespace rồi thoát")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)

    # Cleanup mode
    if args.cleanup:
        ex = KubectlExecutor()
        ex.delete_all_pods()
        sys.exit(0)

    # Tạo service chain giả lập (resource request ngẫu nhiên)
    # Trong thực tế, đây sẽ được đọc từ manifest hoặc API
    np.random.seed(args.seed)
    services = []
    demo_services = [
        ServiceRequest("frontend", 0.03, 128),
        ServiceRequest("auth",     0.03, 128),
        ServiceRequest("catalog",  0.04, 192),
        ServiceRequest("payment",  0.03, 192),
        ServiceRequest("database", 0.05, 256),
    ]

    services = demo_services[:args.services]

    print(f"\n  Service chain sẽ được schedule:")
    for svc in services:
        print(f"    {svc.name}: {svc.cpu_cores} CPU | {svc.mem_mb} MB")

    # Chạy scheduler
    scheduler = RealK8sScheduler(
        model_path=args.model,
        dry_run=args.dry_run,
    )
    scheduler.schedule(services)
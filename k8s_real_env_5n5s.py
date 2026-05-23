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
import csv
from datetime import datetime
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

RESULTS_DIR = "results"
REAL_SERVICE_LOG_CSV = os.path.join(RESULTS_DIR, "real_deployment_services.csv")
REAL_COST_LOG_CSV = os.path.join(RESULTS_DIR, "real_deployment_weighted_cost.csv")


# ═════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeMetrics:
    name: str

    # Actual usage from kubectl top nodes
    cpu_used_cores: float
    mem_used_mb: float

    # Allocatable capacity from kubectl get nodes -o json
    cpu_capacity: float = DEFAULT_NODE_CPU_CAPACITY_CORES
    mem_capacity: float = DEFAULT_NODE_MEM_CAPACITY_MB

    # Allocated resource requests from existing Pods
    cpu_requested_cores: float = 0.0
    mem_requested_mb: float = 0.0

    @property
    def cpu_usage_ratio(self) -> float:
        return min(self.cpu_used_cores / max(self.cpu_capacity, 1e-9), 1.0)

    @property
    def mem_usage_ratio(self) -> float:
        return min(self.mem_used_mb / max(self.mem_capacity, 1e-9), 1.0)

    @property
    def cpu_request_ratio(self) -> float:
        return min(self.cpu_requested_cores / max(self.cpu_capacity, 1e-9), 1.0)

    @property
    def mem_request_ratio(self) -> float:
        return min(self.mem_requested_mb / max(self.mem_capacity, 1e-9), 1.0)

    @property
    def effective_cpu_ratio(self) -> float:
        return max(self.cpu_usage_ratio, self.cpu_request_ratio)

    @property
    def effective_mem_ratio(self) -> float:
        return max(self.mem_usage_ratio, self.mem_request_ratio)

    @property
    def available_cpu_for_requests(self) -> float:
        return max(self.cpu_capacity - self.cpu_requested_cores, 0.0)

    @property
    def available_mem_for_requests(self) -> float:
        return max(self.mem_capacity - self.mem_requested_mb, 0.0)

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

@dataclass
class DependencyEdge:
    src: str
    dst: str
    traffic_weight: float


DEMO_DEPENDENCIES = [
    DependencyEdge("frontend", "auth",     0.90),
    DependencyEdge("frontend", "catalog",  0.80),
    DependencyEdge("catalog",  "payment",  0.70),
    DependencyEdge("payment",  "database", 0.95),
    DependencyEdge("auth",     "database", 0.40),
]

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

    def _parse_resource_cpu_to_cores(self, cpu_value) -> float:
        """
        Parse CPU request:
        - "250m" -> 0.25
        - "1"    -> 1.0
        - None   -> 0.0
        """
        if cpu_value is None:
            return 0.0

        cpu_str = str(cpu_value).strip()

        if cpu_str.endswith("m"):
            return float(cpu_str[:-1]) / 1000.0

        return float(cpu_str)

    def _parse_resource_mem_to_mb(self, mem_value) -> float:
        """
        Parse memory request:
        - "128Mi" -> 128
        - "1Gi"   -> 1024
        - "512Ki" -> 0.5
        - None    -> 0
        """
        if mem_value is None:
            return 0.0

        mem_str = str(mem_value).strip()

        if mem_str.endswith("Ki"):
            return float(mem_str[:-2]) / 1024.0
        if mem_str.endswith("Mi"):
            return float(mem_str[:-2])
        if mem_str.endswith("Gi"):
            return float(mem_str[:-2]) * 1024.0
        if mem_str.endswith("M"):
            return float(mem_str[:-1])
        if mem_str.endswith("G"):
            return float(mem_str[:-1]) * 1024.0

        # fallback: assume bytes
        return float(mem_str) / (1024.0 * 1024.0)

    def _get_allocated_requests_by_node(self) -> dict:
        """
        Tính tổng CPU/RAM requests của các Pod hiện có trên từng node.

        Đây là phần quan trọng để tránh OutOfcpu:
        Kubernetes xét allocated requests, không chỉ xét actual usage.
        """
        allocated = {
            node: {"cpu": 0.0, "mem": 0.0}
            for node in self.worker_nodes
        }

        try:
            data = self._run_kubectl_json(["get", "pods", "-A", "-o", "json"])

            for pod in data.get("items", []):
                spec = pod.get("spec", {})
                status = pod.get("status", {})

                node_name = spec.get("nodeName")
                if node_name not in allocated:
                    continue

                phase = status.get("phase", "")
                if phase in ["Succeeded", "Failed"]:
                    continue

                for container in spec.get("containers", []):
                    resources = container.get("resources", {})
                    requests = resources.get("requests", {})

                    allocated[node_name]["cpu"] += self._parse_resource_cpu_to_cores(
                        requests.get("cpu")
                    )
                    allocated[node_name]["mem"] += self._parse_resource_mem_to_mb(
                        requests.get("memory")
                    )

            return allocated

        except Exception as e:
            print(f"  ⚠️  Không đọc được allocated requests ({e}), dùng 0")
            return allocated

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

        Hàm này kết hợp 2 loại thông tin:
        1. Actual usage từ kubectl top nodes.
        2. Allocated resource requests từ kubectl get pods -A -o json.

        Lý do:
        - kubectl top nodes cho biết CPU/RAM đang dùng thực tế.
        - Kubernetes scheduling lại xét CPU/RAM requests đã được cấp phát.
        - Vì vậy NodeMetrics phải có cả usage và requests để tránh OutOfcpu.
        """

        # Đọc allocated requests trước.
        # Nếu kubectl top bị thiếu node hoặc fail, vẫn còn request info để dùng.
        allocated_requests = self._get_allocated_requests_by_node()

        try:
            result = subprocess.run(
                [
                    "kubectl",
                    "top",
                    "nodes",
                    "--no-headers",
                    "--request-timeout=60s",
                ],
                capture_output=True,
                text=True,
                timeout=60,
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
                    cpu_requested_cores=allocated_requests[name]["cpu"],
                    mem_requested_mb=allocated_requests[name]["mem"],
                )

            metrics = []

            for node in self.worker_nodes:
                cap = self.capacity_map[node]

                # Nếu kubectl top có node này thì dùng actual usage thật.
                # Nếu không có thì fallback actual usage = 0,
                # nhưng vẫn giữ allocated requests.
                if node in metrics_map:
                    metrics.append(metrics_map[node])
                else:
                    metrics.append(
                        NodeMetrics(
                            name=node,
                            cpu_used_cores=0.0,
                            mem_used_mb=0.0,
                            cpu_capacity=cap["cpu"],
                            mem_capacity=cap["mem"],
                            cpu_requested_cores=allocated_requests[node]["cpu"],
                            mem_requested_mb=allocated_requests[node]["mem"],
                        )
                    )

            return metrics

        except Exception as e:
            print(f"  ⚠️  Không đọc được metrics ({e}), dùng actual usage = 0%")

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
                        cpu_requested_cores=allocated_requests[node]["cpu"],
                        mem_requested_mb=allocated_requests[node]["mem"],
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

            actual_cpu_ratio = min(
                (m.cpu_used_cores + extra_cpu) / max(m.cpu_capacity, 1e-9),
                1.0,
            )
            actual_mem_ratio = min(
                (m.mem_used_mb + extra_mem) / max(m.mem_capacity, 1e-9),
                1.0,
            )

            request_cpu_ratio = min(
                (m.cpu_requested_cores + extra_cpu) / max(m.cpu_capacity, 1e-9),
                1.0,
            )
            request_mem_ratio = min(
                (m.mem_requested_mb + extra_mem) / max(m.mem_capacity, 1e-9),
                1.0,
            )

            adj_cpu = max(actual_cpu_ratio, request_cpu_ratio)
            adj_mem = max(actual_mem_ratio, request_mem_ratio)

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
            f"  {'Node':<55} "
            f"{'CPU top':>8} {'CPU req':>8} {'CPU eff':>8} "
            f"{'MEM top':>8} {'MEM req':>8} {'MEM eff':>8} "
            f"{'CPU free':>9}"
        )
        print(f"  {'-'*120}")

        for m in metrics:
            extra_cpu = session_cpu.get(m.name, 0.0)
            extra_mem = session_mem.get(m.name, 0.0)

            cpu_top = min(
                (m.cpu_used_cores + extra_cpu) / max(m.cpu_capacity, 1e-9),
                1.0,
            )
            mem_top = min(
                (m.mem_used_mb + extra_mem) / max(m.mem_capacity, 1e-9),
                1.0,
            )

            cpu_req = min(
                (m.cpu_requested_cores + extra_cpu) / max(m.cpu_capacity, 1e-9),
                1.0,
            )
            mem_req = min(
                (m.mem_requested_mb + extra_mem) / max(m.mem_capacity, 1e-9),
                1.0,
            )

            cpu_eff = max(cpu_top, cpu_req)
            mem_eff = max(mem_top, mem_req)

            cpu_free = max(m.cpu_capacity - m.cpu_requested_cores - extra_cpu, 0.0)

            print(
                f"  {m.name:<55} "
                f"{cpu_top*100:>7.1f}% "
                f"{cpu_req*100:>7.1f}% "
                f"{cpu_eff*100:>7.1f}% "
                f"{mem_top*100:>7.1f}% "
                f"{mem_req*100:>7.1f}% "
                f"{mem_eff*100:>7.1f}% "
                f"{cpu_free:>8.2f}"
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
        Tạo Pod manifest dạng JSON để tránh lỗi YAML indentation.
        kubectl apply -f - chấp nhận cả YAML và JSON.
        """
        cpu_req = f"{int(service.cpu_cores * 1000)}m"
        mem_req = f"{int(service.mem_mb)}Mi"

        pod_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": service.name,
                "namespace": self.namespace,
                "labels": {
                    "app": "drl-placed",
                    "service-name": service.name,
                },
            },
            "spec": {
                "nodeName": node_name,
                "restartPolicy": "Never",
                "containers": [
                    {
                        "name": service.name,
                        "image": service.image,
                        "resources": {
                            "requests": {
                                "cpu": cpu_req,
                                "memory": mem_req,
                            },
                            "limits": {
                                "cpu": cpu_req,
                                "memory": mem_req,
                            },
                        },
                    }
                ],
            },
        }

        return json.dumps(pod_manifest)

    def deploy(self, service: ServiceRequest, node_name: str) -> bool:
        if self.dry_run:
            print(f"    [DRY RUN] {service.name} → {node_name} "
                f"({service.cpu_cores} CPU / {service.mem_mb} MB)")
            return True

        yaml_str = self._make_pod_yaml(service, node_name)

        try:
            result = subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=yaml_str,
                text=True,
                capture_output=True,
                timeout=30,
            )

            if result.returncode != 0:
                print(f"    ❌ kubectl apply failed: {result.stderr.strip()}")
                return False

            print(f"    ✅ Pod object applied: {service.name}")

            # Quan trọng: kiểm tra pod có Ready thật không
            ready = self.wait_for_pod_ready(service.name, timeout_seconds=90)

            if ready:
                actual_node = self.get_pod_node(service.name)
                print(f"    ✅ Pod Ready: {service.name} on {actual_node}")
                return True

            return False

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
    
    def get_pod_status(self, pod_name: str) -> str:
        """Lấy phase/reason hiện tại của pod để debug khi deploy fail."""
        try:
            result = subprocess.run(
                [
                    "kubectl", "get", "pod", pod_name,
                    "-n", self.namespace,
                    "-o", "json"
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return f"Unknown: {result.stderr.strip()}"

            pod = json.loads(result.stdout)
            phase = pod.get("status", {}).get("phase", "Unknown")
            reason = pod.get("status", {}).get("reason", "")

            if reason:
                return f"{phase} / {reason}"

            return phase

        except Exception as e:
            return f"Unknown: {e}"
        
    def wait_for_pod_ready(self, pod_name: str, timeout_seconds: int = 90) -> bool:
        """
        Chờ pod Ready thật sự.

        kubectl apply thành công chỉ có nghĩa object đã được tạo.
        Pod vẫn có thể fail sau đó vì OutOfcpu, ImagePullBackOff, Pending, v.v.
        """
        try:
            result = subprocess.run(
                [
                    "kubectl", "wait",
                    f"pod/{pod_name}",
                    "-n", self.namespace,
                    "--for=condition=Ready",
                    f"--timeout={timeout_seconds}s",
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 10,
            )

            if result.returncode == 0:
                return True

            status = self.get_pod_status(pod_name)
            print(f"    ❌ Pod not Ready: {pod_name} | status={status}")
            print(f"    kubectl wait stderr: {result.stderr.strip()}")
            return False

        except Exception as e:
            status = self.get_pod_status(pod_name)
            print(f"    ❌ Wait error: {e} | pod status={status}")
            return False


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
        self.model_path = model_path
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(RESULTS_DIR, exist_ok=True)
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

    def _get_node_snapshot(
        self,
        metrics: List[NodeMetrics],
        node_name: str,
        session_cpu: dict,
        session_mem: dict,
        ) -> dict:
        """
        Lấy snapshot tài nguyên của node tại thời điểm agent ra quyết định.
        Dùng để log CSV.
        """
        metric_map = {m.name: m for m in metrics}

        if node_name not in metric_map:
            return {
                "cpu_top": "",
                "cpu_req": "",
                "cpu_eff": "",
                "mem_top": "",
                "mem_req": "",
                "mem_eff": "",
                "cpu_free": "",
                "mem_free": "",
            }

        m = metric_map[node_name]

        extra_cpu = session_cpu.get(node_name, 0.0)
        extra_mem = session_mem.get(node_name, 0.0)

        cpu_top = min((m.cpu_used_cores + extra_cpu) / max(m.cpu_capacity, 1e-9), 1.0)
        mem_top = min((m.mem_used_mb + extra_mem) / max(m.mem_capacity, 1e-9), 1.0)

        cpu_req = min((m.cpu_requested_cores + extra_cpu) / max(m.cpu_capacity, 1e-9), 1.0)
        mem_req = min((m.mem_requested_mb + extra_mem) / max(m.mem_capacity, 1e-9), 1.0)

        cpu_eff = max(cpu_top, cpu_req)
        mem_eff = max(mem_top, mem_req)

        cpu_free = max(m.cpu_capacity - m.cpu_requested_cores - extra_cpu, 0.0)
        mem_free = max(m.mem_capacity - m.mem_requested_mb - extra_mem, 0.0)

        return {
            "cpu_top": cpu_top,
            "cpu_req": cpu_req,
            "cpu_eff": cpu_eff,
            "mem_top": mem_top,
            "mem_req": mem_req,
            "mem_eff": mem_eff,
            "cpu_free": cpu_free,
            "mem_free": mem_free,
        }

    def schedule(self, services: List[ServiceRequest]):
        """Chạy scheduling cho toàn bộ service chain."""

        print(f"\n{'═'*55}")
        print(f"  DRL K8s SCHEDULER")
        print(f"  {len(services)} services | "
              f"{'DRY RUN' if self.dry_run else 'LIVE'}")
        print(f"{'═'*55}")

        results = []
        placement_map = {}

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
            node_id = int(action)
            proposed_node = self._node_id_to_name(node_id, metrics)

            print(f"\n     🤖 Agent đề xuất: Node {node_id} ({proposed_node})")

            # 3.5. Feasibility guard — kiểm tra CPU/RAM request thật trước khi deploy
            node_name, fallback_used = self._select_feasible_node(
                proposed_node=proposed_node,
                metrics=metrics,
                service=service,
                session_cpu=session_cpu,
                session_mem=session_mem,
            )

            if fallback_used:
                print(f"     🛡️  Guard đổi node: {proposed_node} → {node_name}")
            else:
                print(f"     ✅ Guard giữ nguyên node: {node_name}")

            node_snapshot = self._get_node_snapshot(
                metrics=metrics,
                node_name=node_name,
                session_cpu=session_cpu,
                session_mem=session_mem,
            )
            # 4. Deploy
            success = self.executor.deploy(service, node_name)
            print(f"     {'✅ Deployed' if success else '❌ Failed'}")

            # 5. Cập nhật session tracking + placement map
            actual_node = ""

            if success:
                session_cpu[node_name] += service.cpu_cores
                session_mem[node_name] += service.mem_mb

                # Trong dry-run, actual_node chính là node_name agent/guard chọn.
                # Trong live-run, đọc lại node thật từ Kubernetes để chắc chắn.
                actual_node = node_name

                if not self.dry_run:
                    pod_node = self.executor.get_pod_node(service.name)
                    if pod_node:
                        actual_node = pod_node

                placement_map[service.name] = actual_node

            self._append_service_log({
                "run_id": self.run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "mode": "dry_run" if self.dry_run else "live",
                "model_path": self.model_path,
                "service_index": idx,
                "service_name": service.name,
                "cpu_request": service.cpu_cores,
                "mem_request_mb": service.mem_mb,
                "proposed_node_id": node_id,
                "proposed_node": proposed_node,
                "final_node": node_name,
                "actual_node": actual_node,
                "fallback_used": fallback_used,
                "success": success,
                "cpu_top": node_snapshot["cpu_top"],
                "cpu_req": node_snapshot["cpu_req"],
                "cpu_eff": node_snapshot["cpu_eff"],
                "mem_top": node_snapshot["mem_top"],
                "mem_req": node_snapshot["mem_req"],
                "mem_eff": node_snapshot["mem_eff"],
                "cpu_free": node_snapshot["cpu_free"],
                "mem_free": node_snapshot["mem_free"],
            })

            results.append({
                "service"     : service.name,
                "node_id"     : node_id,
                "node_name"   : node_name,
                "actual_node" : placement_map.get(service.name, ""),
                "success"     : success,
            })

            # Chờ ngắn để K8s xử lý pod
            if idx < len(services) - 1 and not self.dry_run:
                time.sleep(2)

        self._print_summary(results)

        if placement_map:
            self._print_post_placement_weighted_cost(placement_map)
        else:
            print("\n  ⚠️  Không có placement thành công, bỏ qua weighted cost.")

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

    def _select_feasible_node(
        self,
        proposed_node: str,
        metrics: List[NodeMetrics],
        service: ServiceRequest,
        session_cpu: dict,
        session_mem: dict,
    ) -> tuple[str, bool]:
        """
        Safety layer:
        - Nếu node DQN chọn còn đủ CPU/RAM request thì giữ nguyên.
        - Nếu không đủ, fallback sang node feasible có effective load thấp nhất.
        - Trả về: (chosen_node, fallback_used)
        """
        metric_map = {m.name: m for m in metrics}

        def has_capacity(node_name: str) -> bool:
            m = metric_map[node_name]

            cpu_free = (
                m.cpu_capacity
                - m.cpu_requested_cores
                - session_cpu.get(node_name, 0.0)
            )
            mem_free = (
                m.mem_capacity
                - m.mem_requested_mb
                - session_mem.get(node_name, 0.0)
            )

            return (
                cpu_free >= service.cpu_cores
                and mem_free >= service.mem_mb
            )

        if proposed_node in metric_map and has_capacity(proposed_node):
            return proposed_node, False

        feasible_nodes = [
            node for node in self.worker_nodes
            if has_capacity(node)
        ]

        if not feasible_nodes:
            print(
                f"  ❌ No feasible node for {service.name}: "
                f"requires {service.cpu_cores:.2f} CPU, {service.mem_mb:.0f} MB"
            )
            return proposed_node, False

        def node_score(node_name: str) -> float:
            m = metric_map[node_name]

            cpu_eff = max(
                (m.cpu_used_cores + session_cpu.get(node_name, 0.0)) / max(m.cpu_capacity, 1e-9),
                (m.cpu_requested_cores + session_cpu.get(node_name, 0.0)) / max(m.cpu_capacity, 1e-9),
            )

            mem_eff = max(
                (m.mem_used_mb + session_mem.get(node_name, 0.0)) / max(m.mem_capacity, 1e-9),
                (m.mem_requested_mb + session_mem.get(node_name, 0.0)) / max(m.mem_capacity, 1e-9),
            )

            return cpu_eff + mem_eff

        fallback_node = min(feasible_nodes, key=node_score)

        print(
            f"  ⚠️  Feasibility guard: DQN proposed {proposed_node}, "
            f"but it lacks request capacity. Fallback → {fallback_node}"
        )

        return fallback_node, True
    
    def _node_distance_cost(self, src_node: str, dst_node: str) -> float:
        """
        Logical node-distance cost.

        Vì GKE demo hiện tại chưa đo latency thật giữa pod,
        ta dùng cost logic để phản ánh ý tưởng placement:

        - Cùng node: cost thấp
        - Khác node: cost cao hơn
        """
        if src_node == dst_node:
            return 0.5
        return 2.0
    
    def _print_post_placement_weighted_cost(self, placement_map: dict) -> None:
        """
        Tính post-placement weighted cost dựa trên dependency graph demo.

        Công thức:
            edge_cost = traffic_weight * node_distance_cost(src_node, dst_node)

        Đây không phải latency thật, mà là logical placement cost
        để biểu diễn service-chain dependency sau khi deploy lên GKE.
        """
        print("\n" + "=" * 70)
        print("POST-PLACEMENT WEIGHTED SERVICE-CHAIN COST")
        print("=" * 70)

        total_cost = 0.0
        valid_edges = 0

        print(
            f"{'Edge':<28} {'Weight':>8} {'Src Node':<18} "
            f"{'Dst Node':<18} {'Cost':>8}"
        )
        print("-" * 70)

        for edge in DEMO_DEPENDENCIES:
            if edge.src not in placement_map or edge.dst not in placement_map:
                print(
                    f"{edge.src + ' -> ' + edge.dst:<28} "
                    f"{edge.traffic_weight:>8.2f} "
                    f"{'MISSING':<18} {'MISSING':<18} {'-':>8}"
                )
                continue

            src_node = placement_map[edge.src]
            dst_node = placement_map[edge.dst]

            distance_cost = self._node_distance_cost(src_node, dst_node)
            edge_cost = edge.traffic_weight * distance_cost

            total_cost += edge_cost
            valid_edges += 1

            src_short = src_node.split("-")[-1]
            dst_short = dst_node.split("-")[-1]

            print(
                f"{edge.src + ' -> ' + edge.dst:<28} "
                f"{edge.traffic_weight:>8.2f} "
                f"{src_short:<18} {dst_short:<18} "
                f"{edge_cost:>8.2f}"
            )

        print("-" * 70)
        print(f"Valid dependency edges : {valid_edges}/{len(DEMO_DEPENDENCIES)}")
        print(f"Total weighted cost    : {total_cost:.2f}")
        print("=" * 70)

        self._append_weighted_cost_log(
            placement_map=placement_map,
            total_cost=total_cost,
            valid_edges=valid_edges,
        )

        return total_cost

    def _append_service_log(self, row: dict) -> None:
        """
        Ghi log từng service placement vào CSV.
        """
        fieldnames = [
            "run_id",
            "timestamp",
            "mode",
            "model_path",
            "service_index",
            "service_name",
            "cpu_request",
            "mem_request_mb",
            "proposed_node_id",
            "proposed_node",
            "final_node",
            "actual_node",
            "fallback_used",
            "success",
            "cpu_top",
            "cpu_req",
            "cpu_eff",
            "mem_top",
            "mem_req",
            "mem_eff",
            "cpu_free",
            "mem_free",
        ]

        file_exists = os.path.exists(REAL_SERVICE_LOG_CSV)

        with open(REAL_SERVICE_LOG_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

    def _append_weighted_cost_log(
        self,
        placement_map: dict,
        total_cost: float,
        valid_edges: int,
        ) -> None:
        """
        Ghi dependency weighted cost sau placement vào CSV.
        Mỗi dependency edge là một dòng.
        """
        fieldnames = [
            "run_id",
            "timestamp",
            "mode",
            "model_path",
            "src",
            "dst",
            "traffic_weight",
            "src_node",
            "dst_node",
            "distance_cost",
            "edge_cost",
            "valid_edges",
            "total_cost",
        ]

        file_exists = os.path.exists(REAL_COST_LOG_CSV)

        with open(REAL_COST_LOG_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for edge in DEMO_DEPENDENCIES:
                if edge.src not in placement_map or edge.dst not in placement_map:
                    continue

                src_node = placement_map[edge.src]
                dst_node = placement_map[edge.dst]

                distance_cost = self._node_distance_cost(src_node, dst_node)
                edge_cost = edge.traffic_weight * distance_cost

                writer.writerow({
                    "run_id": self.run_id,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "mode": "dry_run" if self.dry_run else "live",
                    "model_path": self.model_path,
                    "src": edge.src,
                    "dst": edge.dst,
                    "traffic_weight": edge.traffic_weight,
                    "src_node": src_node,
                    "dst_node": dst_node,
                    "distance_cost": distance_cost,
                    "edge_cost": edge_cost,
                    "valid_edges": valid_edges,
                    "total_cost": total_cost,
                })
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
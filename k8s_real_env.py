# k8s_real_env.py
"""
Bridge layer: kết nối DRL agent đã train với Kubernetes cluster thật.

Model DQN đã train theo dynamic env 8 nodes / 10 services.
Khi test thật với cluster 5 nodes / 5 services, observation vẫn được pad về 208-dim.

Flow file-to-file:
1. provision_gke.ps1
   -> xuất cluster_state.json

2. generate_latency_test_yaml.py
   -> đọc cluster_state.json
   -> sinh latency-test.yaml

3. measure_latency.ps1
   -> đo ping giữa các pod netshoot
   -> xuất real_node_latency_matrix.json

4. k8s_real_env.py
   -> đọc cluster_state.json để lấy node order
   -> đọc real_node_latency_matrix.json để tính chain latency thật
   -> schedule service bằng DQN + action masking
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import subprocess
import time
import json
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from envs.topology import create_sample_service_chain, SERVICE_PROFILES


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

WORKER_NODES: List[str] = []

# Fallback capacity nếu không đọc được từ kubectl.
NODE_CPU_CAPACITY_CORES = 0.94
NODE_MEM_CAPACITY_MB = 2800.0

# Test thật 5 worker nodes / 5 services.
NUM_NODES = 5
NUM_SERVICES = 5

# Model train dynamic 8 nodes / 10 services.
OBS_MAX_NODES = 8
OBS_MAX_SERVICES = 10
OBS_NODE_FEAT_DIM = 5
OBS_SERVICE_FEAT_DIM = 4

OBS_OBS_DIM = (
    OBS_MAX_NODES * OBS_NODE_FEAT_DIM +
    OBS_MAX_SERVICES * OBS_SERVICE_FEAT_DIM +
    (OBS_MAX_SERVICES * OBS_MAX_SERVICES) +
    OBS_MAX_SERVICES +
    OBS_MAX_NODES +
    OBS_MAX_SERVICES
)

K8S_NAMESPACE = "drl-scheduler"

# Vì bạn thường chạy từ thư mục scripts:
#   python ..\k8s_real_env.py ...
# nên default path là file trong thư mục scripts hiện tại.
DEFAULT_CLUSTER_STATE_PATH = "cluster_state.json"
DEFAULT_LATENCY_MATRIX_PATH = "real_node_latency_matrix.json"


# ═════════════════════════════════════════════════════════════════════════════
# NODE DISCOVERY
# ═════════════════════════════════════════════════════════════════════════════

def discover_worker_nodes(
    expected_count: int = NUM_NODES,
    cluster_state_path: str = DEFAULT_CLUSTER_STATE_PATH,
) -> List[str]:
    """
    Ưu tiên đọc node order từ cluster_state.json.

    Nếu không có file thì fallback sang đọc node labels từ kubectl:
        Node 0: small
        Node 1: medium
        Node 2: large
        Node 3: medium
        Node 4: small
    """

    path = Path(cluster_state_path)

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        node_order = sorted(
            state["node_order"],
            key=lambda x: int(x["index"]),
        )

        ordered_nodes = [n["name"] for n in node_order]

        print(f"  Loaded worker node order from {cluster_state_path}:")
        for n in node_order:
            print(f"    Node {n['index']} ({n['role']}) -> {n['name']}")

        if len(ordered_nodes) < expected_count:
            raise RuntimeError(
                f"{cluster_state_path} has {len(ordered_nodes)} nodes, "
                f"need {expected_count}."
            )

        return ordered_nodes[:expected_count]

    print(f"  ⚠️  {cluster_state_path} not found. Fallback to kubectl labels.")

    def get_nodes_by_size(size: str) -> List[str]:
        result = subprocess.run(
            [
                "kubectl", "get", "nodes",
                "-l", f"node-size={size}",
                "-o", "jsonpath={range .items[*]}{.metadata.name}{\"\\n\"}{end}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip()
                or f"kubectl get nodes node-size={size} failed"
            )

        nodes = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        ]

        return sorted(nodes)

    small_nodes = get_nodes_by_size("small")
    medium_nodes = get_nodes_by_size("medium")
    large_nodes = get_nodes_by_size("large")

    if len(small_nodes) >= 2 and len(medium_nodes) >= 2 and len(large_nodes) >= 1:
        ordered_nodes = [
            small_nodes[0],    # Node 0: small
            medium_nodes[0],   # Node 1: medium
            large_nodes[0],    # Node 2: large
            medium_nodes[1],   # Node 3: medium
            small_nodes[1],    # Node 4: small
        ]

        print("  Worker node order mapped to training topology:")
        for i, node_name in enumerate(ordered_nodes):
            print(f"    Node {i} -> {node_name}")

        return ordered_nodes[:expected_count]

    raise RuntimeError(
        "Cannot discover worker nodes. Need cluster_state.json or node-size labels."
    )


# ═════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeMetrics:
    name: str
    cpu_used_cores: float
    mem_used_mb: float
    cpu_capacity: float = NODE_CPU_CAPACITY_CORES
    mem_capacity: float = NODE_MEM_CAPACITY_MB

    @property
    def cpu_util(self) -> float:
        if self.cpu_capacity <= 0:
            return 1.0
        return min(self.cpu_used_cores / self.cpu_capacity, 1.0)

    @property
    def mem_util(self) -> float:
        if self.mem_capacity <= 0:
            return 1.0
        return min(self.mem_used_mb / self.mem_capacity, 1.0)


@dataclass
class ServiceRequest:
    name: str
    cpu_cores: float
    mem_mb: float
    service_type: str = "balanced"
    placed_on: int = -1
    image: str = "nginx:alpine"


# ═════════════════════════════════════════════════════════════════════════════
# REAL CLUSTER OBSERVER
# ═════════════════════════════════════════════════════════════════════════════

class RealClusterObserver:
    """Đọc CPU/RAM usage thật của worker nodes qua kubectl."""

    def __init__(self, worker_nodes: List[str]):
        self.worker_nodes = worker_nodes
        self.node_capacities = self._load_node_capacities()

    def _parse_cpu(self, cpu_str: str) -> float:
        cpu_str = cpu_str.strip()

        if cpu_str.endswith("m"):
            return float(cpu_str[:-1]) / 1000.0

        if cpu_str.endswith("n"):
            return float(cpu_str[:-1]) / 1_000_000_000.0

        return float(cpu_str)

    def _parse_mem(self, mem_str: str) -> float:
        mem_str = mem_str.strip()

        if mem_str.endswith("Ki"):
            return float(mem_str[:-2]) / 1024.0

        if mem_str.endswith("Mi"):
            return float(mem_str[:-2])

        if mem_str.endswith("Gi"):
            return float(mem_str[:-2]) * 1024.0

        return float(mem_str) / (1024.0 * 1024.0)

    def _load_node_capacities(self) -> dict:
        capacities = {}

        for node_name in self.worker_nodes:
            try:
                result = subprocess.run(
                    [
                        "kubectl", "get", "node", node_name,
                        "-o", "jsonpath={.status.allocatable.cpu} {.status.allocatable.memory}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip())

                parts = result.stdout.strip().split()

                if len(parts) < 2:
                    raise RuntimeError(
                        f"Unexpected capacity output for {node_name}: {result.stdout!r}"
                    )

                cpu_capacity = self._parse_cpu(parts[0])
                mem_capacity = self._parse_mem(parts[1])

                capacities[node_name] = (cpu_capacity, mem_capacity)

            except Exception as e:
                print(f"  ⚠️  Không đọc được allocatable của {node_name}: {e}")
                print("      Dùng fallback capacity.")
                capacities[node_name] = (
                    NODE_CPU_CAPACITY_CORES,
                    NODE_MEM_CAPACITY_MB,
                )

        return capacities

    def get_metrics(self) -> List[NodeMetrics]:
        """
        Parse output kubectl top nodes.

        Ví dụ:
            node-name   27m   1%   177Mi   6%
        """

        try:
            result = subprocess.run(
                ["kubectl", "top", "nodes", "--no-headers"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip())

            metrics_map = {}

            for line in result.stdout.strip().splitlines():
                parts = line.split()

                if len(parts) < 4:
                    continue

                name = parts[0]

                if name not in self.worker_nodes:
                    continue

                cpu_cores = self._parse_cpu(parts[1])
                mem_mb = self._parse_mem(parts[3])

                cpu_capacity, mem_capacity = self.node_capacities.get(
                    name,
                    (NODE_CPU_CAPACITY_CORES, NODE_MEM_CAPACITY_MB),
                )

                metrics_map[name] = NodeMetrics(
                    name=name,
                    cpu_used_cores=cpu_cores,
                    mem_used_mb=mem_mb,
                    cpu_capacity=cpu_capacity,
                    mem_capacity=mem_capacity,
                )

            return [
                metrics_map.get(
                    n,
                    NodeMetrics(
                        name=n,
                        cpu_used_cores=0.0,
                        mem_used_mb=0.0,
                        cpu_capacity=self.node_capacities.get(
                            n,
                            (NODE_CPU_CAPACITY_CORES, NODE_MEM_CAPACITY_MB),
                        )[0],
                        mem_capacity=self.node_capacities.get(
                            n,
                            (NODE_CPU_CAPACITY_CORES, NODE_MEM_CAPACITY_MB),
                        )[1],
                    ),
                )
                for n in self.worker_nodes
            ]

        except Exception as e:
            print(f"  ⚠️  Không đọc được metrics ({e}), dùng 0% usage.")

            return [
                NodeMetrics(
                    name=n,
                    cpu_used_cores=0.0,
                    mem_used_mb=0.0,
                    cpu_capacity=self.node_capacities.get(
                        n,
                        (NODE_CPU_CAPACITY_CORES, NODE_MEM_CAPACITY_MB),
                    )[0],
                    mem_capacity=self.node_capacities.get(
                        n,
                        (NODE_CPU_CAPACITY_CORES, NODE_MEM_CAPACITY_MB),
                    )[1],
                )
                for n in self.worker_nodes
            ]

    def build_observation(
        self,
        metrics: List[NodeMetrics],
        service: ServiceRequest,
        service_idx: int,
        services: List[ServiceRequest],
        dependency_edges: list = None,
        session_cpu: dict = None,
        session_mem: dict = None,
    ) -> np.ndarray:
        """
        Chuyển metrics thật thành obs vector 208-dim theo format train.
        """

        obs = []

        session_cpu = session_cpu or {}
        session_mem = session_mem or {}
        dependency_edges = dependency_edges or []

        max_cpu_cap = max((m.cpu_capacity for m in metrics), default=1.0)
        max_mem_cap = max((m.mem_capacity for m in metrics), default=1.0)

        if max_cpu_cap <= 0:
            max_cpu_cap = 1.0

        if max_mem_cap <= 0:
            max_mem_cap = 1.0

        # Node features.
        for i in range(OBS_MAX_NODES):
            if i < len(metrics):
                m = metrics[i]

                extra_cpu = session_cpu.get(m.name, 0.0)
                extra_mem = session_mem.get(m.name, 0.0)

                used_cpu = min(m.cpu_used_cores + extra_cpu, m.cpu_capacity)
                used_mem = min(m.mem_used_mb + extra_mem, m.mem_capacity)

                available_cpu_norm = float(
                    np.clip((m.cpu_capacity - used_cpu) / max_cpu_cap, 0.0, 1.0)
                )
                available_mem_norm = float(
                    np.clip((m.mem_capacity - used_mem) / max_mem_cap, 0.0, 1.0)
                )

                cpu_cap_norm = float(
                    np.clip(m.cpu_capacity / max_cpu_cap, 0.0, 1.0)
                )
                mem_cap_norm = float(
                    np.clip(m.mem_capacity / max_mem_cap, 0.0, 1.0)
                )

                is_active = 1.0

                obs.extend(
                    [
                        available_cpu_norm,
                        available_mem_norm,
                        cpu_cap_norm,
                        mem_cap_norm,
                        is_active,
                    ]
                )

            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # Service features.
        profile_names = list(SERVICE_PROFILES.keys())
        num_types = len(profile_names)

        for i in range(OBS_MAX_SERVICES):
            if i < len(services):
                svc = services[i]

                cpu_req_norm = float(np.clip(svc.cpu_cores / 2.0, 0.0, 1.0))
                mem_req_norm = float(np.clip(svc.mem_mb / 4096.0, 0.0, 1.0))

                if svc.service_type in profile_names:
                    type_idx = profile_names.index(svc.service_type)
                else:
                    type_idx = 0

                service_type_id_norm = float(type_idx) / float(max(1, num_types))
                is_placed = 1.0 if svc.placed_on >= 0 else 0.0

                obs.extend(
                    [
                        cpu_req_norm,
                        mem_req_norm,
                        service_type_id_norm,
                        is_placed,
                    ]
                )

            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        # Dependency matrix.
        dep_mat = np.zeros(
            (OBS_MAX_SERVICES, OBS_MAX_SERVICES),
            dtype=np.float32,
        )

        for edge in dependency_edges:
            if edge.src < OBS_MAX_SERVICES and edge.dst < OBS_MAX_SERVICES:
                dep_mat[edge.src][edge.dst] = float(edge.traffic_weight)

        obs.extend(list(dep_mat.reshape(-1)))

        # Current service onehot.
        current_onehot = [0.0] * OBS_MAX_SERVICES

        if service_idx < OBS_MAX_SERVICES:
            current_onehot[service_idx] = 1.0

        obs.extend(current_onehot)

        # Valid node mask.
        valid_node_count = min(len(metrics), OBS_MAX_NODES)
        valid_node_mask = (
            [1.0] * valid_node_count +
            [0.0] * (OBS_MAX_NODES - valid_node_count)
        )
        obs.extend(valid_node_mask)

        # Valid service mask.
        valid_service_count = min(len(services), OBS_MAX_SERVICES)
        valid_service_mask = (
            [1.0] * valid_service_count +
            [0.0] * (OBS_MAX_SERVICES - valid_service_count)
        )
        obs.extend(valid_service_mask)

        arr = np.array(obs, dtype=np.float32)

        if arr.shape != (OBS_OBS_DIM,):
            raise RuntimeError(
                f"Observation shape mismatch: got {arr.shape}, expected {(OBS_OBS_DIM,)}"
            )

        return arr

    def print_cluster_state(
        self,
        metrics: List[NodeMetrics],
        session_cpu: dict = None,
        session_mem: dict = None,
    ):
        session_cpu = session_cpu or {}
        session_mem = session_mem or {}

        print(
            f"  {'Node':<45} {'CPU(adj)':>10} {'MEM(adj)':>10} "
            f"{'CPU cap':>10} {'MEM cap':>10}"
        )
        print(f"  {'-' * 92}")

        for m in metrics:
            extra_cpu = session_cpu.get(m.name, 0.0)
            extra_mem = session_mem.get(m.name, 0.0)

            adj_cpu = min(
                (m.cpu_used_cores + extra_cpu) / m.cpu_capacity,
                1.0,
            ) if m.cpu_capacity > 0 else 1.0

            adj_mem = min(
                (m.mem_used_mb + extra_mem) / m.mem_capacity,
                1.0,
            ) if m.mem_capacity > 0 else 1.0

            print(
                f"  {m.name:<45} "
                f"{adj_cpu * 100:>8.1f}% "
                f"{adj_mem * 100:>8.1f}% "
                f"{m.cpu_capacity:>8.2f} "
                f"{m.mem_capacity:>8.0f}Mi"
            )


# ═════════════════════════════════════════════════════════════════════════════
# KUBECTL EXECUTOR
# ═════════════════════════════════════════════════════════════════════════════

class KubectlExecutor:
    """Deploy pod lên đúng node qua kubectl."""

    def __init__(self, namespace: str = K8S_NAMESPACE, dry_run: bool = False):
        self.namespace = namespace
        self.dry_run = dry_run

        if not dry_run:
            self._ensure_namespace()

    def _ensure_namespace(self):
        subprocess.run(
            ["kubectl", "create", "namespace", self.namespace],
            capture_output=True,
            text=True,
        )

    def _make_pod_yaml(self, service: ServiceRequest, node_name: str) -> str:
        cpu_req = f"{int(service.cpu_cores * 1000)}m"
        mem_req = f"{int(service.mem_mb)}Mi"

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
            print(
                f"    [DRY RUN] {service.name} -> {node_name} "
                f"({service.cpu_cores:.3f} CPU / {service.mem_mb:.0f} Mi)"
            )
            return True

        yaml_str = self._make_pod_yaml(service, node_name)

        try:
            result = subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=yaml_str,
                text=True,
                capture_output=True,
                timeout=20,
            )

            if result.returncode != 0:
                print("    ❌ kubectl apply failed:")
                print(result.stderr.strip())
                return False

            print(result.stdout.strip())
            return True

        except Exception as e:
            print(f"    ❌ Deploy error: {e}")
            return False

    def delete_all_pods(self):
        if self.dry_run:
            print(f"  [DRY RUN] Delete all pods in {self.namespace}")
            return

        subprocess.run(
            [
                "kubectl", "delete", "pods", "--all",
                "-n", self.namespace,
                "--wait=false",
            ],
            capture_output=True,
            text=True,
        )

        print(f"  Đã xóa tất cả pods trong namespace '{self.namespace}'")


# ═════════════════════════════════════════════════════════════════════════════
# REAL K8S SCHEDULER
# ═════════════════════════════════════════════════════════════════════════════

class RealK8sScheduler:
    """
    Main scheduler:
    - Load DQN model
    - Đọc real cluster
    - Build observation 208-dim
    - Model chọn node bằng Q-values + action masking
    - Feasibility guard kiểm tra CPU/RAM
    - Deploy pod bằng nodeName
    - Tính traffic-weighted chain latency bằng latency thật nếu có
    """

    def __init__(
        self,
        model_path: str,
        dry_run: bool = False,
        worker_nodes: Optional[List[str]] = None,
        resource_scale: float = 1.0,
        debug_print_obs: bool = False,
        cluster_state_path: str = DEFAULT_CLUSTER_STATE_PATH,
        latency_matrix_path: str = DEFAULT_LATENCY_MATRIX_PATH,
    ):
        self.dry_run = dry_run
        self.cluster_state_path = cluster_state_path
        self.latency_matrix_path = latency_matrix_path

        self.worker_nodes = worker_nodes or discover_worker_nodes(
            cluster_state_path=self.cluster_state_path
        )

        global WORKER_NODES
        WORKER_NODES = list(self.worker_nodes)

        self.observer = RealClusterObserver(self.worker_nodes)
        self.executor = KubectlExecutor(dry_run=dry_run)

        self.resource_scale = float(resource_scale)
        self.debug_print_obs = bool(debug_print_obs)

        self.real_latency_matrix = self._load_real_latency_matrix(
            self.latency_matrix_path
        )

        from stable_baselines3 import DQN

        print(f"\n  Đang load model: {model_path} ...")
        self.model = DQN.load(model_path)
        print("  Model loaded")

        try:
            print(f"  Model observation space: {self.model.observation_space}")
            print(f"  Model action space     : {self.model.action_space}")
        except Exception:
            pass

        self.dependency_edges = []

    def _load_real_latency_matrix(self, latency_matrix_path: str):
        path = Path(latency_matrix_path)

        if not path.exists():
            print(
                f"  ⚠️  {latency_matrix_path} not found. "
                "Using estimated latency: same-node=0ms, cross-node=1ms."
            )
            return None

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            matrix = data.get("latency_ms")

            if matrix is None:
                print(
                    f"  ⚠️  latency_ms not found in {latency_matrix_path}. "
                    "Using estimated latency."
                )
                return None

            print(f"  Loaded real latency matrix from {latency_matrix_path}")
            return matrix

        except Exception as e:
            print(
                f"  ⚠️  Failed to load {latency_matrix_path}: {e}. "
                "Using estimated latency."
            )
            return None

    def make_service_chain(self, service_count: int, seed: int) -> List[ServiceRequest]:
        chain = create_sample_service_chain(service_count, seed=seed)
        self.dependency_edges = chain.get_dependencies()

        services: List[ServiceRequest] = []

        for svc in chain.services:
            scaled_cpu = float(svc.cpu_request) * self.resource_scale
            scaled_mem = float(svc.memory_request * 1024.0) * self.resource_scale

            services.append(
                ServiceRequest(
                    name=svc.name,
                    cpu_cores=scaled_cpu,
                    mem_mb=scaled_mem,
                    service_type=svc.service_type,
                    image="nginx:alpine",
                )
            )

        return services

    def _node_id_to_name(self, node_id: int, metrics: List[NodeMetrics]) -> str:
        if 0 <= node_id < len(WORKER_NODES):
            return WORKER_NODES[node_id]

        print(f"  ⚠️  node_id={node_id} out of range after masking, fallback least-loaded")

        return min(
            WORKER_NODES,
            key=lambda n: next(
                (m.cpu_util for m in metrics if m.name == n),
                1.0,
            ),
        )

    def _predict_valid_action(
        self,
        obs: np.ndarray,
        valid_node_count: int,
        debug: bool = False,
    ) -> int:
        """
        Predict action cho DQN nhưng mask các padding nodes.

        Model train với action space Discrete(8), nhưng real cluster có thể chỉ có 5 node.
        Vì vậy action hợp lệ chỉ là [0, valid_node_count - 1].
        Các action padding như 5,6,7 sẽ bị set Q-value = -inf trước khi argmax.
        """

        import torch

        obs_tensor = torch.as_tensor(
            obs.reshape(1, -1),
            dtype=torch.float32,
            device=self.model.device,
        )

        with torch.no_grad():
            q_values = self.model.q_net(obs_tensor).cpu().numpy()[0]

        raw_best_action = int(np.argmax(q_values))

        masked_q_values = q_values.copy()

        if valid_node_count < len(masked_q_values):
            masked_q_values[valid_node_count:] = -1e9

        masked_action = int(np.argmax(masked_q_values))

        if debug:
            print(f"    [DEBUG] raw DQN action before mask = {raw_best_action}")
            print(f"    [DEBUG] valid_node_count = {valid_node_count}")
            print(
                "    [DEBUG] q_values raw =",
                np.array2string(q_values, precision=3, separator=", ")
            )
            print(
                "    [DEBUG] q_values masked =",
                np.array2string(masked_q_values, precision=3, separator=", ")
            )
            print(f"    [DEBUG] masked action = {masked_action}")

        return masked_action

    def _find_feasible_node(
        self,
        service: ServiceRequest,
        preferred_node: str,
        metrics: List[NodeMetrics],
        session_cpu: dict,
        session_mem: dict,
    ) -> Optional[str]:

        def remaining(name: str):
            metric = next((m for m in metrics if m.name == name), None)

            if metric is None:
                return None

            used_cpu = metric.cpu_used_cores + session_cpu.get(name, 0.0)
            used_mem = metric.mem_used_mb + session_mem.get(name, 0.0)

            return metric.cpu_capacity - used_cpu, metric.mem_capacity - used_mem

        preferred_remaining = remaining(preferred_node)

        if preferred_remaining is not None:
            rem_cpu, rem_mem = preferred_remaining

            if rem_cpu >= service.cpu_cores and rem_mem >= service.mem_mb:
                return preferred_node

        feasible_nodes = []

        for name in WORKER_NODES:
            rem = remaining(name)

            if rem is None:
                continue

            rem_cpu, rem_mem = rem

            if rem_cpu >= service.cpu_cores and rem_mem >= service.mem_mb:
                metric = next((m for m in metrics if m.name == name), None)

                if metric is None:
                    projected_score = float("inf")
                else:
                    proj_cpu = (
                        metric.cpu_used_cores +
                        session_cpu.get(name, 0.0) +
                        service.cpu_cores
                    ) / metric.cpu_capacity

                    proj_mem = (
                        metric.mem_used_mb +
                        session_mem.get(name, 0.0) +
                        service.mem_mb
                    ) / metric.mem_capacity

                    projected_score = max(proj_cpu, proj_mem)

                feasible_nodes.append((projected_score, name))

        if feasible_nodes:
            feasible_nodes.sort(key=lambda item: item[0])
            return feasible_nodes[0][1]

        return None

    def _print_capacity_diagnosis(
        self,
        service: ServiceRequest,
        metrics: List[NodeMetrics],
        session_cpu: dict,
        session_mem: dict,
    ):
        print("     Resource diagnosis:")
        print(
            f"     Service request: "
            f"{service.cpu_cores:.3f} CPU | {service.mem_mb:.0f} Mi"
        )

        best_cpu = float("-inf")
        best_mem = float("-inf")

        for m in metrics:
            used_cpu = m.cpu_used_cores + session_cpu.get(m.name, 0.0)
            used_mem = m.mem_used_mb + session_mem.get(m.name, 0.0)

            rem_cpu = m.cpu_capacity - used_cpu
            rem_mem = m.mem_capacity - used_mem

            best_cpu = max(best_cpu, rem_cpu)
            best_mem = max(best_mem, rem_mem)

            print(
                f"       {m.name}: remaining "
                f"{rem_cpu:.3f} CPU | {rem_mem:.0f} Mi "
                f"(capacity {m.cpu_capacity:.3f} CPU | {m.mem_capacity:.0f} Mi)"
            )

        if best_cpu == float("-inf"):
            best_cpu = 0.0

        if best_mem == float("-inf"):
            best_mem = 0.0

        print(f"     Best remaining: {best_cpu:.3f} CPU | {best_mem:.0f} Mi")

    # ═════════════════════════════════════════════════════════════════════════
    # CHAIN LATENCY METRICS
    # ═════════════════════════════════════════════════════════════════════════

    def _estimate_node_latency_ms(self, src_node: int, dst_node: int) -> float:
        """
        Lấy latency thật từ real_node_latency_matrix.json nếu có.

        Fallback:
        - same node = 0 ms
        - cross node = 1 ms
        """
        if src_node == dst_node:
            return 0.0

        if self.real_latency_matrix is not None:
            try:
                return float(self.real_latency_matrix[src_node][dst_node])
            except Exception:
                pass

        return 1.0

    def _calculate_chain_latency_metrics(self, services: List[ServiceRequest]) -> dict:
        """
        Tính traffic-weighted chain latency dựa trên dependency_edges.

        Formula:
            total = sum(latency(node(src), node(dst)) * traffic_weight)
        """

        if not self.dependency_edges:
            return {
                "total_weighted_latency": 0.0,
                "total_edges": 0,
                "same_node_edges": 0,
                "cross_node_edges": 0,
                "total_weight": 0.0,
                "same_node_weight": 0.0,
                "cross_node_weight": 0.0,
                "weighted_colocation_rate": 0.0,
                "edge_details": [],
            }

        total_weighted_latency = 0.0
        total_weight = 0.0

        same_node_edges = 0
        cross_node_edges = 0

        same_node_weight = 0.0
        cross_node_weight = 0.0

        edge_details = []

        for edge in self.dependency_edges:
            if edge.src >= len(services) or edge.dst >= len(services):
                continue

            src_svc = services[edge.src]
            dst_svc = services[edge.dst]

            if src_svc.placed_on < 0 or dst_svc.placed_on < 0:
                continue

            src_node = src_svc.placed_on
            dst_node = dst_svc.placed_on

            latency_ms = self._estimate_node_latency_ms(src_node, dst_node)
            traffic_weight = float(edge.traffic_weight)
            weighted_latency = latency_ms * traffic_weight

            total_weighted_latency += weighted_latency
            total_weight += traffic_weight

            same_node = src_node == dst_node

            if same_node:
                same_node_edges += 1
                same_node_weight += traffic_weight
            else:
                cross_node_edges += 1
                cross_node_weight += traffic_weight

            edge_details.append(
                {
                    "src": src_svc.name,
                    "dst": dst_svc.name,
                    "src_node": src_node,
                    "dst_node": dst_node,
                    "same_node": same_node,
                    "latency_ms": latency_ms,
                    "traffic_weight": traffic_weight,
                    "weighted_latency": weighted_latency,
                }
            )

        weighted_colocation_rate = (
            same_node_weight / total_weight
            if total_weight > 0
            else 0.0
        )

        return {
            "total_weighted_latency": total_weighted_latency,
            "total_edges": len(edge_details),
            "same_node_edges": same_node_edges,
            "cross_node_edges": cross_node_edges,
            "total_weight": total_weight,
            "same_node_weight": same_node_weight,
            "cross_node_weight": cross_node_weight,
            "weighted_colocation_rate": weighted_colocation_rate,
            "edge_details": edge_details,
        }

    def _print_chain_latency_metrics(self, services: List[ServiceRequest]):
        metrics = self._calculate_chain_latency_metrics(services)

        latency_source = (
            "real_node_latency_matrix.json"
            if self.real_latency_matrix is not None
            else "estimated same-node=0ms, cross-node=1ms"
        )

        print(f"\n{'═' * 55}")
        print("  CHAIN LATENCY METRICS")
        print(f"{'═' * 55}")

        print(f"  Latency source                : {latency_source}")
        print(f"  Total dependency edges        : {metrics['total_edges']}")
        print(f"  Same-node edges               : {metrics['same_node_edges']}")
        print(f"  Cross-node edges              : {metrics['cross_node_edges']}")
        print(f"  Total traffic weight          : {metrics['total_weight']:.3f}")
        print(f"  Same-node traffic weight      : {metrics['same_node_weight']:.3f}")
        print(f"  Cross-node traffic weight     : {metrics['cross_node_weight']:.3f}")
        print(f"  Weighted co-location rate     : {metrics['weighted_colocation_rate'] * 100:.1f}%")
        print(f"  Traffic-weighted latency      : {metrics['total_weighted_latency']:.3f} ms")

        print("\n  Edge details:")
        print(
            f"  {'SRC':<20} {'DST':<20} {'Nodes':<12} "
            f"{'Weight':>8} {'Latency':>10} {'Weighted':>10}"
        )
        print(f"  {'-' * 86}")

        for e in metrics["edge_details"]:
            node_pair = f"{e['src_node']}->{e['dst_node']}"
            print(
                f"  {e['src']:<20} "
                f"{e['dst']:<20} "
                f"{node_pair:<12} "
                f"{e['traffic_weight']:>8.3f} "
                f"{e['latency_ms']:>8.3f}ms "
                f"{e['weighted_latency']:>8.3f}"
            )

        print(f"{'═' * 55}")

    def schedule(self, services: List[ServiceRequest]):
        print(f"\n{'═' * 55}")
        print("  DRL K8s SCHEDULER")
        print(f"  {len(services)} services | {'DRY RUN' if self.dry_run else 'LIVE'}")
        print(f"{'═' * 55}")

        results = []

        session_cpu = {n: 0.0 for n in WORKER_NODES}
        session_mem = {n: 0.0 for n in WORKER_NODES}

        for idx, service in enumerate(services):
            print(f"\n  ── Service {idx + 1}/{len(services)}: {service.name} ──")
            print(
                f"     Yêu cầu: "
                f"{service.cpu_cores:.3f} CPU | {service.mem_mb:.0f} Mi RAM"
            )

            metrics = self.observer.get_metrics()
            self.observer.print_cluster_state(metrics, session_cpu, session_mem)

            obs = self.observer.build_observation(
                metrics=metrics,
                service=service,
                service_idx=idx,
                services=services,
                dependency_edges=self.dependency_edges,
                session_cpu=session_cpu,
                session_mem=session_mem,
            )

            if self.debug_print_obs:
                print("\n    [DEBUG] obs shape:", obs.shape)
                print(
                    "    [DEBUG] node features obs[:40] =",
                    np.array2string(obs[:40], precision=3, separator=", "),
                )

            node_id = self._predict_valid_action(
                obs=obs,
                valid_node_count=len(WORKER_NODES),
                debug=self.debug_print_obs,
            )

            node_name = self._node_id_to_name(node_id, metrics)

            feasible_node = self._find_feasible_node(
                service=service,
                preferred_node=node_name,
                metrics=metrics,
                session_cpu=session_cpu,
                session_mem=session_mem,
            )

            if feasible_node is None:
                print(f"\n     Agent chọn sau masking: Node {node_id} ({node_name})")
                print("     Không có node nào còn đủ CPU/RAM cho service này.")
                self._print_capacity_diagnosis(
                    service,
                    metrics,
                    session_cpu,
                    session_mem,
                )

                results.append(
                    {
                        "service": service.name,
                        "node_id": node_id,
                        "node_name": node_name,
                        "success": False,
                    }
                )

                if idx < len(services) - 1 and not self.dry_run:
                    time.sleep(2)

                continue

            if feasible_node != node_name:
                print(f"\n     Agent chọn sau masking: Node {node_id} ({node_name})")
                print(f"     Fallback sang node còn đủ tài nguyên: {feasible_node}")
                node_name = feasible_node
            else:
                print(f"\n     Agent chọn sau masking: Node {node_id} ({node_name})")

            success = self.executor.deploy(service, node_name)
            print(f"     {'Deployed' if success else 'Failed'}")

            if success:
                session_cpu[node_name] += service.cpu_cores
                session_mem[node_name] += service.mem_mb
                service.placed_on = (
                    WORKER_NODES.index(node_name)
                    if node_name in WORKER_NODES
                    else node_id
                )

            results.append(
                {
                    "service": service.name,
                    "node_id": node_id,
                    "node_name": node_name,
                    "success": success,
                }
            )

            if idx < len(services) - 1 and not self.dry_run:
                time.sleep(2)

        self._print_summary(results)
        self._print_chain_latency_metrics(services)

        return results

    def _print_summary(self, results: list):
        n_ok = sum(1 for r in results if r["success"])

        print(f"\n{'═' * 55}")
        print(f"  KẾT QUẢ: {n_ok}/{len(results)} services scheduled")
        print(f"{'═' * 55}")
        print(f"  {'Service':<20} {'Node':<45} Status")
        print(f"  {'-' * 76}")

        for r in results:
            icon = "OK" if r["success"] else "FAILED"
            print(f"  {r['service']:<20} {r['node_name']:<45} {icon}")

        print(f"{'═' * 55}")

        if n_ok > 0 and not self.dry_run:
            print("\n  Xem pods đã deploy:")
            print(f"  kubectl get pods -n {K8S_NAMESPACE} -o wide")
            print("\n  Xóa sau khi xem xong:")
            print(f"  kubectl delete pods --all -n {K8S_NAMESPACE}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real K8s Scheduler dùng DQN dynamic 8x10"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="./models/dqn_model",
        help="Path đến model DQN, không cần .zip",
    )

    parser.add_argument(
        "--services",
        type=int,
        default=5,
        help="Số services trong chain",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chạy thử, không deploy thật",
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Xóa tất cả pods trong namespace rồi thoát",
    )

    parser.add_argument(
        "--nodes",
        type=str,
        default="",
        help="Danh sách worker nodes phân tách bằng dấu phẩy; bỏ trống để tự phát hiện",
    )

    parser.add_argument(
        "--resource-scale",
        type=float,
        default=1.0,
        help="Scale factor cho CPU/RAM request của service",
    )

    parser.add_argument(
        "--debug-print-obs",
        action="store_true",
        help="In observation vector, raw Q-values và masked action debug",
    )

    parser.add_argument(
        "--cluster-state",
        type=str,
        default=DEFAULT_CLUSTER_STATE_PATH,
        help="Path to cluster_state.json exported by provision_gke.ps1",
    )

    parser.add_argument(
        "--latency-matrix",
        type=str,
        default=DEFAULT_LATENCY_MATRIX_PATH,
        help="Path to real_node_latency_matrix.json exported by measure_latency.ps1",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)

    if args.cleanup:
        executor = KubectlExecutor()
        executor.delete_all_pods()
        sys.exit(0)

    worker_nodes = [
        n.strip()
        for n in args.nodes.split(",")
        if n.strip()
    ] or None

    scheduler = RealK8sScheduler(
        model_path=args.model,
        dry_run=args.dry_run,
        worker_nodes=worker_nodes,
        resource_scale=args.resource_scale,
        debug_print_obs=args.debug_print_obs,
        cluster_state_path=args.cluster_state,
        latency_matrix_path=args.latency_matrix,
    )

    services = scheduler.make_service_chain(args.services, args.seed)

    print("\n  Service chain sẽ được schedule:")
    for svc in services:
        print(
            f"    {svc.name}: "
            f"type={svc.service_type} | "
            f"{svc.cpu_cores:.3f} CPU | "
            f"{svc.mem_mb:.0f} Mi"
        )

    scheduler.schedule(services)
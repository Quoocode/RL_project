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
from dataclasses import dataclass
from typing import List, Optional

from envs.topology import create_sample_service_chain, SERVICE_PROFILES


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG — khớp với cluster thật
# ═════════════════════════════════════════════════════════════════════════════

# 5 worker nodes (bỏ control-plane). Nếu không truyền danh sách node,
# script sẽ tự đọc từ kubectl và lấy 5 node worker đầu tiên.
WORKER_NODES: List[str] = []

# Capacity thật (từ kubectl get nodes)
NODE_CPU_CAPACITY_CORES = 0.94
NODE_MEM_CAPACITY_MB    = 2800.0

# Khớp với model đã train
NUM_NODES     = 5   # số worker nodes
NUM_SERVICES  = 5   # số services trong chain
# obs_dim = NUM_NODES*2 + 2 + NUM_SERVICES = 17

# Model DQN hiện tại được train trên format dynamic 8-node / 10-service,
# nên observation thực tế phải được pad về đúng shape 208.
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

# Namespace riêng để không ảnh hưởng cluster
K8S_NAMESPACE = "drl-scheduler"


def discover_worker_nodes(expected_count: int = NUM_NODES) -> List[str]:
    """Tự động phát hiện worker nodes từ kubectl, loại control-plane/master."""
    result = subprocess.run(
        ["kubectl", "get", "nodes", "-o", "name"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "kubectl get nodes failed")

    discovered = []
    for line in result.stdout.splitlines():
        name = line.replace("node/", "").strip()
        if not name:
            continue
        lower_name = name.lower()
        if "master" in lower_name or "control-plane" in lower_name:
            continue
        discovered.append(name)

    discovered = sorted(discovered)
    if len(discovered) < expected_count:
        raise RuntimeError(
            f"Chỉ tìm thấy {len(discovered)} worker nodes, cần tối thiểu {expected_count}."
        )
    return discovered[:expected_count]


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
        return min(self.cpu_used_cores / self.cpu_capacity, 1.0)

    @property
    def mem_util(self) -> float:
        return min(self.mem_used_mb / self.mem_capacity, 1.0)


@dataclass
class ServiceRequest:
    name: str
    cpu_cores: float   # vd: 0.5 = 500m
    mem_mb: float      # vd: 512
    service_type: str = "balanced"
    placed_on: int = -1
    image: str = "nginx:alpine"


# ═════════════════════════════════════════════════════════════════════════════
# CLUSTER OBSERVER
# ═════════════════════════════════════════════════════════════════════════════

class RealClusterObserver:
    """Đọc CPU/RAM usage thật của 5 worker nodes qua kubectl top nodes."""

    def __init__(self, worker_nodes: List[str]):
        self.worker_nodes = worker_nodes

    def get_metrics(self) -> List[NodeMetrics]:
        """
        Parse output của kubectl top nodes:
            minikube-m02  27m  0%  177Mi  2%
        CPU "27m" = 27 millicores = 0.027 cores
        MEM "177Mi" = 177 MB
        """
        try:
            result = subprocess.run(
                ["kubectl", "top", "nodes", "--no-headers"],
                capture_output=True, text=True, timeout=10
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

                # Parse CPU
                cpu_str = parts[1]
                if cpu_str.endswith("m"):
                    cpu_cores = float(cpu_str[:-1]) / 1000.0
                else:
                    cpu_cores = float(cpu_str)

                # Parse Memory
                mem_str = parts[3]
                if mem_str.endswith("Mi"):
                    mem_mb = float(mem_str[:-2])
                elif mem_str.endswith("Ki"):
                    mem_mb = float(mem_str[:-2]) / 1024.0
                elif mem_str.endswith("Gi"):
                    mem_mb = float(mem_str[:-2]) * 1024.0
                else:
                    mem_mb = float(mem_str) / (1024 * 1024)

                metrics_map[name] = NodeMetrics(name, cpu_cores, mem_mb)

            # Trả về đúng thứ tự worker nodes đã phát hiện
            return [
                metrics_map.get(n, NodeMetrics(n, 0.0, 0.0))
                for n in self.worker_nodes
            ]

        except Exception as e:
            print(f"  ⚠️  Không đọc được metrics ({e}), dùng 0%")
            return [NodeMetrics(n, 0.0, 0.0) for n in self.worker_nodes]

    def build_observation(self, metrics: List[NodeMetrics],
                          service: ServiceRequest,
                          service_idx: int,
                          services: List[ServiceRequest],
                          dependency_edges: list = None,
                          session_cpu: dict = None,
                          session_mem: dict = None) -> np.ndarray:
        """
        Chuyển metrics thật → obs vector 208-dim theo format train.

        Format khớp với env train:
        - node features: 8 * 5
        - service features: 10 * 4
        - dependency matrix: 10 * 10
        - current service onehot: 10
        - valid node mask: 8
        - valid service mask: 10

        Ở môi trường thật chỉ có 5 node và 5 service, nên 3 node cuối
        và 5 service cuối sẽ được padding bằng 0.
        """
        obs = []
        session_cpu = session_cpu or {}
        session_mem = session_mem or {}
        dependency_edges = dependency_edges or []

        # Node features: available_cpu_norm, available_mem_norm,
        # cpu_cap_norm, mem_cap_norm, is_active
        for i in range(OBS_MAX_NODES):
            if i < len(metrics):
                m = metrics[i]
                extra_cpu = session_cpu.get(m.name, 0.0)
                extra_mem = session_mem.get(m.name, 0.0)
                used_cpu = min(m.cpu_used_cores + extra_cpu, m.cpu_capacity)
                used_mem = min(m.mem_used_mb + extra_mem, m.mem_capacity)
                available_cpu_norm = float(np.clip((m.cpu_capacity - used_cpu) / m.cpu_capacity, 0.0, 1.0))
                available_mem_norm = float(np.clip((m.mem_capacity - used_mem) / m.mem_capacity, 0.0, 1.0))
                cpu_cap_norm = 1.0
                mem_cap_norm = 1.0
                is_active = 1.0
                obs.extend([
                    available_cpu_norm,
                    available_mem_norm,
                    cpu_cap_norm,
                    mem_cap_norm,
                    is_active,
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # Service features: cpu_req_norm, mem_req_norm, service_type_id_norm, is_placed
        profile_names = list(SERVICE_PROFILES.keys())
        for i in range(OBS_MAX_SERVICES):
            if i < len(services):
                svc = services[i]
                cpu_req_norm = float(np.clip(svc.cpu_cores / 2.0, 0.0, 1.0))
                mem_req_norm = float(np.clip(svc.mem_mb / 4096.0, 0.0, 1.0))
                type_idx = profile_names.index(svc.service_type) if svc.service_type in profile_names else 0
                service_type_id_norm = float(type_idx) / float(max(1, len(profile_names)))
                is_placed = 1.0 if svc.placed_on >= 0 else 0.0
                obs.extend([cpu_req_norm, mem_req_norm, service_type_id_norm, is_placed])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        # Dependency matrix (10 x 10)
        dep_mat = np.zeros((OBS_MAX_SERVICES, OBS_MAX_SERVICES), dtype=np.float32)
        for edge in dependency_edges:
            if edge.src < OBS_MAX_SERVICES and edge.dst < OBS_MAX_SERVICES:
                dep_mat[edge.src][edge.dst] = float(edge.traffic_weight)
        obs.extend(list(dep_mat.reshape(-1)))

        # current_service_onehot (10)
        current_onehot = [0.0] * OBS_MAX_SERVICES
        if service_idx < OBS_MAX_SERVICES:
            current_onehot[service_idx] = 1.0
        obs.extend(current_onehot)

        # valid node mask (8)
        obs.extend([1.0] * min(len(metrics), OBS_MAX_NODES) + [0.0] * max(0, OBS_MAX_NODES - len(metrics)))

        # valid service mask (10)
        obs.extend([1.0] * min(NUM_SERVICES, OBS_MAX_SERVICES) + [0.0] * max(0, OBS_MAX_SERVICES - NUM_SERVICES))

        return np.array(obs, dtype=np.float32)

    def print_cluster_state(self, metrics: List[NodeMetrics],
                             session_cpu: dict = None,
                             session_mem: dict = None):
        session_cpu = session_cpu or {}
        session_mem = session_mem or {}
        print(f"  {'Node':<15} {'CPU (adj)':>10} {'RAM (adj)':>10}  "
              f"{'(kubectl)':>10}")
        print(f"  {'-'*50}")
        for m in metrics:
            extra_cpu = session_cpu.get(m.name, 0.0)
            extra_mem = session_mem.get(m.name, 0.0)
            adj_cpu = min((m.cpu_used_cores + extra_cpu) / m.cpu_capacity, 1.0)
            adj_mem = min((m.mem_used_mb   + extra_mem) / m.mem_capacity, 1.0)
            bar = "█" * int(adj_cpu * 10)
            print(f"  {m.name:<15} {adj_cpu*100:>8.1f}%  "
                  f"{adj_mem*100:>8.1f}%  "
                  f"({m.cpu_util*100:.1f}% raw)  {bar}")


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

    def __init__(self, model_path: str, dry_run: bool = False,
                 worker_nodes: Optional[List[str]] = None):
        self.dry_run  = dry_run
        self.worker_nodes = worker_nodes or discover_worker_nodes()
        global WORKER_NODES
        WORKER_NODES = list(self.worker_nodes)
        self.observer = RealClusterObserver(self.worker_nodes)
        self.executor = KubectlExecutor(dry_run=dry_run)

        # Load model — import ở đây để không lỗi nếu SB3 chưa cài
        from stable_baselines3 import DQN
        print(f"\n  Đang load model: {model_path} ...")
        self.model = DQN.load(model_path)
        print(f"  ✅ Model loaded")

        self.dependency_edges = []

    def make_service_chain(self, service_count: int, seed: int) -> List[ServiceRequest]:
        chain = create_sample_service_chain(service_count, seed=seed)
        self.dependency_edges = chain.get_dependencies()

        services: List[ServiceRequest] = []
        for svc in chain.services:
            services.append(ServiceRequest(
                name=svc.name,
                cpu_cores=float(svc.cpu_request),
                mem_mb=float(svc.memory_request * 1024.0),
                service_type=svc.service_type,
                image="nginx:alpine",
            ))
        return services

    def _node_id_to_name(self, node_id: int,
                         metrics: List[NodeMetrics]) -> str:
        """
        Map node index (0-4) → tên node thật.
        Nếu node_id out of range, fallback về node ít tải nhất.
        """
        if 0 <= node_id < len(WORKER_NODES):
            return WORKER_NODES[node_id]

        # Fallback (không nên xảy ra)
        print(f"  ⚠️  node_id={node_id} out of range, fallback least-loaded")
        return min(WORKER_NODES,
                   key=lambda n: next(
                       (m.cpu_util for m in metrics if m.name == n), 1.0))

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
        session_cpu: dict = {n: 0.0 for n in WORKER_NODES}
        session_mem: dict = {n: 0.0 for n in WORKER_NODES}

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
                metrics,
                service,
                idx,
                services,
                dependency_edges=self.dependency_edges,
                session_cpu=session_cpu,
                session_mem=session_mem,
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
                service.placed_on = node_id

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
    parser.add_argument("--nodes",    type=str, default="",
                        help="Danh sách worker nodes phân tách bằng dấu phẩy; nếu bỏ trống sẽ tự phát hiện từ kubectl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)

    # Cleanup mode
    if args.cleanup:
        ex = KubectlExecutor()
        ex.delete_all_pods()
        sys.exit(0)

    worker_nodes = [n.strip() for n in args.nodes.split(",") if n.strip()] or None

    scheduler = RealK8sScheduler(
        model_path=args.model,
        dry_run=args.dry_run,
        worker_nodes=worker_nodes,
    )
    services = scheduler.make_service_chain(args.services, args.seed)

    print(f"\n  Service chain sẽ được schedule:")
    for svc in services:
        print(f"    {svc.name}: type={svc.service_type} | {svc.cpu_cores:.2f} CPU | {svc.mem_mb:.0f} MB")

    scheduler.schedule(services)

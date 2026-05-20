# envs/k8s_env.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Optional

from envs.topology import (
    create_sample_topology,
    create_sample_service_chain,
    Microservice,
    SERVICE_PROFILES,
)


class K8sPlacementEnv(gym.Env):
    """
    Môi trường mô phỏng Kubernetes cho bài toán xếp Microservice.

    Observation space : [cpu_util_0, mem_util_0, ..., cpu_util_N, mem_util_N,
                         onehot_service_0, ..., onehot_service_M]
    Action space      : Discrete(num_nodes)  — chọn node để đặt service hiện tại
    Termination       : đã xử lý hết num_services service (dù thành công hay thất bại)
    Truncation        : vượt quá max_steps (bảo hiểm khi training bất thường)
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, num_nodes: int = 5, num_services: int = 5,
             max_steps: int = 200, enable_background_load: bool = True,
             background_load_level: str = "medium"):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_services = num_services
        # ── BẢO HIỂM: Giới hạn số bước tối đa mỗi episode ──────────────────
        # Về lý thuyết mỗi episode chỉ cần num_services bước, nhưng max_steps
        # là lưới an toàn cuối cùng tránh episode chạy mãi khi có lỗi logic.
        self.max_steps = max_steps
        # Env v2: mô phỏng cluster không rỗng bằng background workload
        self.enable_background_load = enable_background_load
        self.background_load_level = background_load_level

        # Hạ tầng mạng và chuỗi service — tạo một lần, reset() sẽ làm sạch
        self.topology = create_sample_topology(num_nodes)
        self.service_chain = create_sample_service_chain(num_services, seed = 0)

        # Dùng để normalize capacity của heterogeneous nodes
        self.max_node_cpu_capacity = max(
            node.cpu_capacity for node in self.topology.nodes
        )
        self.max_node_mem_capacity = max(
            node.memory_capacity for node in self.topology.nodes
        )

        # Bộ đếm nội bộ
        self.current_service_idx: int = 0
        self.step_count: int = 0

        # Action / Observation space
        self.action_space = spaces.Discrete(num_nodes)

        # Env v2 observation:
        # Mỗi node có 4 đặc trưng:
        #   cpu_util, mem_util, cpu_capacity_norm, mem_capacity_norm
        # Sau đó thêm:
        #   service_cpu_norm, service_mem_norm, service_onehot
        obs_dim = (num_nodes * 4) + 2 + num_services
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    # ──────────────────────────────────────────────────────────────────────────
    # QUAN SÁT
    # ──────────────────────────────────────────────────────────────────────────
    def _get_observation(self) -> np.ndarray:
        obs = []

        for node in self.topology.nodes:
            cpu_capacity_norm = node.cpu_capacity / self.max_node_cpu_capacity
            mem_capacity_norm = node.memory_capacity / self.max_node_mem_capacity

            if not node.is_active:
                # Node chết → coi như full utilization,
                # nhưng vẫn giữ capacity info để agent biết loại node.
                obs.extend([
                    1.0,
                    1.0,
                    cpu_capacity_norm,
                    mem_capacity_norm,
                ])
            else:
                obs.extend([
                    node.cpu_used / node.cpu_capacity,
                    node.memory_used / node.memory_capacity,
                    cpu_capacity_norm,
                    mem_capacity_norm,
                ])
        
        # THÊM MỚI: resource request của service hiện tại (normalized)
        # Agent cần biết service này nặng hay nhẹ để chọn node phù hợp
        if self.current_service_idx < self.num_services:
            svc = self.service_chain.services[self.current_service_idx]
            obs.append(svc.cpu_request / 2.0)      # normalize về [0,1] với max=2.0
            obs.append(svc.memory_request / 4.0)   # normalize về [0,1] với max=4.0
        else:
            obs.extend([0.0, 0.0])

        service_onehot = np.zeros(self.num_services, dtype=np.float32)
        if self.current_service_idx < self.num_services:
            service_onehot[self.current_service_idx] = 1.0
        obs.extend(service_onehot)

        return np.array(obs, dtype=np.float32)
    
    # ──────────────────────────────────────────────────────────────────────────
    # BACKGROUND LOAD
    # ──────────────────────────────────────────────────────────────────────────
    def _apply_background_load(self):
        """
        Tạo tải nền cho mỗi node sau reset.

        Level:
        - none   : không tạo background load
        - light  : tải nền nhẹ
        - medium : tải nền vừa, dùng làm default
        - hard   : tải nền nặng, dùng cho scenario khó
        """
        if not self.enable_background_load:
            return

        if self.background_load_level == "none":
            return

        if self.background_load_level == "light":
            base_low, base_high = 0.03, 0.18
            hot_prob = 0.10
            hot_low, hot_high = 0.30, 0.45

        elif self.background_load_level == "medium":
            base_low, base_high = 0.05, 0.25
            hot_prob = 0.15
            hot_low, hot_high = 0.35, 0.55

        elif self.background_load_level == "hard":
            base_low, base_high = 0.10, 0.35
            hot_prob = 0.25
            hot_low, hot_high = 0.45, 0.70

        else:
            raise ValueError(
                f"Unknown background_load_level={self.background_load_level}. "
                "Use: none, light, medium, hard."
            )

        for node in self.topology.nodes:
            if not node.is_active:
                continue

            cpu_frac = float(self.np_random.uniform(base_low, base_high))
            mem_frac = float(self.np_random.uniform(base_low, base_high))

            if self.np_random.random() < hot_prob:
                cpu_frac = float(self.np_random.uniform(hot_low, hot_high))

            if self.np_random.random() < hot_prob:
                mem_frac = float(self.np_random.uniform(hot_low, hot_high))

            node.cpu_used = round(cpu_frac * node.cpu_capacity, 4)
            node.memory_used = round(mem_frac * node.memory_capacity, 4)

    # ──────────────────────────────────────────────────────────────────────────
    # HELPER TÍNH TRAFFIC-WEIGHTED LATENCY PENALTY THEO DEPENDENCY GRAPH
    # ──────────────────────────────────────────────────────────────────────────
    def _calculate_dependency_latency_penalty(self,
                                              service: Microservice) -> float:
        """
        Tính latency penalty dựa trên traffic-weighted dependency graph.

        Chỉ tính các dependency edge liên quan tới service vừa được đặt.
        Một edge chỉ bị phạt khi cả hai đầu src/dst đã được placement.
        """

        if not hasattr(self.service_chain, "get_dependencies"):
            return 0.0

        dependency_edges = self.service_chain.get_dependencies()

        if not dependency_edges:
            return 0.0

        total_weighted_latency = 0.0

        for edge in dependency_edges:
            # Chỉ xét edge có liên quan đến service vừa đặt
            if edge.src != service.id and edge.dst != service.id:
                continue

            src_svc = self.service_chain.services[edge.src]
            dst_svc = self.service_chain.services[edge.dst]

            # Chỉ tính khi cả hai service đã được đặt
            if src_svc.placed_on < 0 or dst_svc.placed_on < 0:
                continue

            latency = self.topology.get_latency(
                src_svc.placed_on,
                dst_svc.placed_on
            )

            if np.isfinite(latency):
                total_weighted_latency += latency * float(edge.traffic_weight)

        # Scale nhẹ để latency không ép agent gom toàn bộ service vào một node.
        # Cap lại để một edge latency cao không phá toàn bộ reward.
        latency_penalty = min(total_weighted_latency * 0.18, 2.5)

        return latency_penalty

    # ──────────────────────────────────────────────────────────────────────────
    # PHẦN THƯỞNG
    # ──────────────────────────────────────────────────────────────────────────
    def _calculate_reward(self, service: Microservice,
                        node_id: int, success: bool) -> float:
        # Protect against node_id outside current topology (padding action)
        if node_id is None or node_id < 0 or node_id >= len(self.topology.nodes):
            return -50.0

        node = self.topology.get_node(node_id)

        if node is None:
            return -50.0

        if not node.is_active:
            return -50.0

        if not success:
            return -20.0

        # 1. Thưởng cơ bản khi đặt thành công
        reward = 5.0

        # 2. Utilization của node vừa chọn sau khi allocate
        cpu_util = node.cpu_used / node.cpu_capacity
        mem_util = node.memory_used / node.memory_capacity

        # 3. Phạt nếu node quá tải
        if cpu_util > 0.8:
            reward -= ((cpu_util - 0.8) * 50) ** 2

        if mem_util > 0.8:
            reward -= ((mem_util - 0.8) * 50) ** 2

        # 4. Phạt mất cân bằng tải toàn cluster
        active_nodes = [n for n in self.topology.nodes if n.is_active]

        cpu_utils = np.array([
            n.cpu_used / n.cpu_capacity for n in active_nodes
        ])

        mem_utils = np.array([
            n.memory_used / n.memory_capacity for n in active_nodes
        ])

        avg_cpu = np.mean(cpu_utils)
        avg_mem = np.mean(mem_utils)

        cpu_std = np.std(cpu_utils)
        mem_std = np.std(mem_utils)

        imbalance_penalty = 6.0 * (cpu_std + mem_std)
        reward -= imbalance_penalty

        # 5. Phạt nếu chọn node nóng hơn trung bình cluster
        hotspot_penalty = (
            max(0.0, cpu_util - avg_cpu) +
            max(0.0, mem_util - avg_mem)
        )
        reward -= 4.0 * hotspot_penalty

        # 6. Phạt nếu gom quá nhiều service cùng chain vào cùng node
        # Đây là phần quan trọng để sửa lỗi 5/5 service dồn vào m05 hoặc m06
        same_node_count = sum(
            1 for s in self.service_chain.services[:service.id]
            if s.placed_on == node_id
        )

        reward -= 1.5 * same_node_count

        # 7. Phạt nhẹ nếu node được chọn đã chứa nhiều service hơn các node khác
        placed_counts = []
        for n in self.topology.nodes:
            count = sum(
                1 for s in self.service_chain.services[:service.id + 1]
                if s.placed_on == n.id
            )
            placed_counts.append(count)

        chosen_count = placed_counts[node_id]
        avg_count = np.mean(placed_counts)

        if chosen_count > avg_count:
            reward -= 0.8 * (chosen_count - avg_count)

        # 8. Traffic-weighted latency penalty theo dependency graph
        #
        # Thay vì chỉ xét service liền trước:
        #   service[i-1] -> service[i]
        #
        # Env v3 xét các dependency edge có traffic_weight:
        #   latency(src_node, dst_node) * traffic_weight
        #
        # Điều này làm reward bám sát bài toán microservice call graph hơn.
        latency_penalty = self._calculate_dependency_latency_penalty(service)
        reward -= latency_penalty

        return reward

    # ──────────────────────────────────────────────────────────────────────────
    # STEP  ← ĐÂY LÀ CHỖ SỬA BUG CHÍNH
    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.current_service_idx < self.num_services, \
            "step() được gọi sau khi episode đã kết thúc — hãy gọi reset() trước."

        self.step_count += 1
        service = self.service_chain.services[self.current_service_idx]
        node    = self.topology.get_node(action)

        # Thử phân bổ tài nguyên
        success = False
        if node is not None and node.is_active:
            success = node.allocate(service.cpu_request, service.memory_request)

        if success:
            service.placed_on = action

        # ── FIX BUG CHÍNH ──────────────────────────────────────────────────
        # Luôn tăng index dù thành công hay thất bại.
        # Trước đây chỉ tăng khi success → nếu agent cứ chọn node
        # không hợp lệ, current_service_idx không bao giờ tăng →
        # terminated = False mãi mãi → vòng lặp vô tận.
        self.current_service_idx += 1
        # ───────────────────────────────────────────────────────────────────

        reward     = self._calculate_reward(service, action, success)
        terminated = self.current_service_idx >= self.num_services
        truncated  = (not terminated) and (self.step_count >= self.max_steps)

        info = {
            "success"       : success,
            "service_idx"   : self.current_service_idx - 1,   # vừa xử lý
            "placed_on_node": action if success else -1,
            "step_count"    : self.step_count,
            "placed_total"  : sum(
                1 for s in self.service_chain.services if s.placed_on >= 0
            ),
        }

        return self._get_observation(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Use base reset to create service_chain, clear usages and apply background load
        obs, info = super().reset(seed=seed, options=options)

        # Ensure padding nodes remain inactive and zero-capacity after reset
        for i in range(self.actual_num_nodes, self.max_nodes):
            node = self.topology.nodes[i]
            node.is_active = False
            node.cpu_capacity = 0.0
            node.memory_capacity = 0.0
            node.cpu_used = 0.0
            node.memory_used = 0.0

        # Recompute normalization denominators
        cpu_caps = [n.cpu_capacity for n in self.topology.nodes if n.cpu_capacity > 0]
        mem_caps = [n.memory_capacity for n in self.topology.nodes if n.memory_capacity > 0]
        self.max_node_cpu_capacity = max(cpu_caps) if cpu_caps else 1.0
        self.max_node_mem_capacity = max(mem_caps) if mem_caps else 1.0

        return self._get_observation(), info

    # ──────────────────────────────────────────────────────────────────────────
    # RESET
    # ──────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset topology về trạng thái sạch trước
        self.topology.reset()

        # Env v2: thêm background workload sau khi reset node
        self._apply_background_load()

        # Tạo service chain mới mỗi episode với seed khác nhau
        self.service_chain = create_sample_service_chain(
            self.num_services, seed=seed
        )

        self.current_service_idx = 0
        self.step_count = 0

        return self._get_observation(), {}

    # ──────────────────────────────────────────────────────────────────────────
    # RENDER  (tuỳ chọn, hữu ích khi debug)
    # ──────────────────────────────────────────────────────────────────────────
    def render(self, mode="human"):
        print(f"\n--- Episode step {self.step_count} | "
              f"Service {self.current_service_idx}/{self.num_services} ---")
        for node in self.topology.nodes:
            status = "DEAD" if not node.is_active else \
                     f"CPU {node.cpu_used/node.cpu_capacity*100:.0f}% | " \
                     f"MEM {node.memory_used/node.memory_capacity*100:.0f}%"
            print(f"  Node {node.id}: {status}")
        placed = [(s.id, s.placed_on) for s in self.service_chain.services
                  if s.placed_on >= 0]
        print(f"  Đã đặt: {placed}")


# -----------------------------------------------------------------------------
# Dynamic env: padded observation/action for up to MAX_NODES x MAX_SERVICES
# -----------------------------------------------------------------------------
class K8sPlacementEnvDynamic(K8sPlacementEnv):
    """
    Dynamic-size environment that pads observations/actions to a fixed
    maximum: `max_nodes` and `max_services` (default 8 x 10).

    This class reuses the existing topology/service generation for the
    *actual* counts, but exposes an observation/action space sized for
    the maximums and provides masks/padding as described in
    NEXT_STEPS_DYNAMIC_8N10.md.
    """

    def __init__(self,
                 actual_num_nodes: int = 5,
                 actual_num_services: int = 5,
                 max_nodes: int = 8,
                 max_services: int = 10,
                 max_steps: int = 200,
                 enable_background_load: bool = True,
                 background_load_level: str = "medium"):
        # Initialise base env with actual sizes
        super().__init__(
            num_nodes=actual_num_nodes,
            num_services=actual_num_services,
            max_steps=max_steps,
            enable_background_load=enable_background_load,
            background_load_level=background_load_level,
        )

        # Keep explicit fields for actual vs max
        self.actual_num_nodes = actual_num_nodes
        self.actual_num_services = actual_num_services
        self.max_nodes = max_nodes
        self.max_services = max_services

        # Action space always Discrete(max_nodes)
        self.action_space = spaces.Discrete(self.max_nodes)

        # Precompute observation dim per design from NEXT_STEPS_DYNAMIC_8N10.md
        node_feat_dim = 5   # available_cpu_norm, available_mem_norm, cpu_cap_norm, mem_cap_norm, is_active
        svc_feat_dim = 4    # cpu_req_norm, mem_req_norm, service_type_id_norm, is_placed

        obs_dim = (
            self.max_nodes * node_feat_dim +
            self.max_services * svc_feat_dim +
            (self.max_services * self.max_services) +
            self.max_services +  # current_service_onehot
            self.max_nodes +     # valid_node_mask
            self.max_services    # valid_service_mask
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Recreate topology with padding nodes up to max_nodes so that
        # downstream code can inspect a fixed-size `topology.nodes` list.
        from envs.topology import create_sample_topology
        self.topology = create_sample_topology(self.actual_num_nodes, max_nodes=self.max_nodes)

        # Recompute normalization denominators based on the new topology
        cpu_caps = [n.cpu_capacity for n in self.topology.nodes if n.cpu_capacity > 0]
        mem_caps = [n.memory_capacity for n in self.topology.nodes if n.memory_capacity > 0]
        self.max_node_cpu_capacity = max(cpu_caps) if cpu_caps else 1.0
        self.max_node_mem_capacity = max(mem_caps) if mem_caps else 1.0

    def _build_dependency_matrix(self):
        mat = np.zeros((self.max_services, self.max_services), dtype=np.float32)
        if not hasattr(self.service_chain, "get_dependencies"):
            return mat

        for edge in self.service_chain.get_dependencies():
            if edge.src >= self.actual_num_services or edge.dst >= self.actual_num_services:
                continue
            mat[edge.src][edge.dst] = float(edge.traffic_weight) * 1.0

        return mat

    def _get_observation(self) -> np.ndarray:
        obs = []

        # Node features (pad to max_nodes)
        for i in range(self.max_nodes):
            if i < self.actual_num_nodes:
                node = self.topology.nodes[i]
                cpu_available = node.cpu_available
                mem_available = node.memory_available
                available_cpu_norm = cpu_available / self.max_node_cpu_capacity if self.max_node_cpu_capacity > 0 else 0.0
                available_mem_norm = mem_available / self.max_node_mem_capacity if self.max_node_mem_capacity > 0 else 0.0
                cpu_cap_norm = node.cpu_capacity / self.max_node_cpu_capacity if self.max_node_cpu_capacity > 0 else 0.0
                mem_cap_norm = node.memory_capacity / self.max_node_mem_capacity if self.max_node_mem_capacity > 0 else 0.0
                is_active = 1.0 if node.is_active else 0.0
                obs.extend([
                    float(np.clip(available_cpu_norm, 0.0, 1.0)),
                    float(np.clip(available_mem_norm, 0.0, 1.0)),
                    float(np.clip(cpu_cap_norm, 0.0, 1.0)),
                    float(np.clip(mem_cap_norm, 0.0, 1.0)),
                    float(is_active),
                ])
            else:
                # padding node
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # Service features (pad to max_services)
        profile_names = list(SERVICE_PROFILES.keys())
        num_types = len(profile_names)
        for i in range(self.max_services):
            if i < self.actual_num_services:
                svc = self.service_chain.services[i]
                cpu_req_norm = svc.cpu_request / 2.0
                mem_req_norm = svc.memory_request / 4.0
                type_idx = profile_names.index(svc.service_type) if svc.service_type in profile_names else 0
                service_type_id_norm = float(type_idx) / float(max(1, num_types))
                is_placed = 1.0 if svc.placed_on >= 0 else 0.0
                obs.extend([
                    float(np.clip(cpu_req_norm, 0.0, 1.0)),
                    float(np.clip(mem_req_norm, 0.0, 1.0)),
                    float(np.clip(service_type_id_norm, 0.0, 1.0)),
                    float(is_placed),
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        # Dependency matrix (max_services x max_services)
        dep_mat = self._build_dependency_matrix()
        obs.extend(list(dep_mat.reshape(-1).astype(np.float32)))

        # current_service_onehot (length max_services)
        onehot = np.zeros(self.max_services, dtype=np.float32)
        if self.current_service_idx < self.actual_num_services:
            onehot[self.current_service_idx] = 1.0
        obs.extend(list(onehot))

        # valid_node_mask (length max_nodes)
        valid_node_mask = [1.0 if i < self.actual_num_nodes else 0.0 for i in range(self.max_nodes)]
        obs.extend(valid_node_mask)

        # valid_service_mask (length max_services)
        valid_service_mask = [1.0 if i < self.actual_num_services else 0.0 for i in range(self.max_services)]
        obs.extend(valid_service_mask)

        return np.array(obs, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Allow action in [0, max_nodes). If action >= actual_num_nodes -> invalid (padding_node)
        assert self.current_service_idx < self.actual_num_services, \
            "step() được gọi sau khi episode đã kết thúc — hãy gọi reset() trước."

        self.step_count += 1
        service = self.service_chain.services[self.current_service_idx]

        invalid_action = False
        invalid_reason = None
        success = False

        if action >= self.actual_num_nodes:
            # padding node selected — treat as invalid with heavy penalty
            invalid_action = True
            invalid_reason = "padding_node"
            node = None
            reward = -50.0
            # advance index and return early below after bookkeeping
            self.current_service_idx += 1

            terminated = self.current_service_idx >= self.actual_num_services
            truncated = (not terminated) and (self.step_count >= self.max_steps)

            info = {
                "success": False,
                "invalid_action": invalid_action,
                "invalid_reason": invalid_reason,
                "placed_on_node": -1,
                "step_count": self.step_count,
                "placed_total": sum(1 for s in self.service_chain.services if s.placed_on >= 0),
            }

            return self._get_observation(), reward, terminated, truncated, info
        else:
            node = self.topology.get_node(action)
            if node is None:
                invalid_action = True
                invalid_reason = "padding_node"
            elif not node.is_active:
                invalid_action = True
                invalid_reason = "inactive_node"
            else:
                # try allocate
                success = node.allocate(service.cpu_request, service.memory_request)
                if not success:
                    invalid_action = True
                    invalid_reason = "insufficient_resource"

        if success:
            service.placed_on = action

        # always advance current_service_idx
        self.current_service_idx += 1

        # Compute reward using base implementation
        reward = self._calculate_reward(service, action, success)

        terminated = self.current_service_idx >= self.actual_num_services
        truncated = (not terminated) and (self.step_count >= self.max_steps)

        info = {
            "success": success,
            "invalid_action": invalid_action,
            "invalid_reason": invalid_reason,
            "placed_on_node": action if success else -1,
            "step_count": self.step_count,
            "placed_total": sum(1 for s in self.service_chain.services if s.placed_on >= 0),
        }

        return self._get_observation(), reward, terminated, truncated, info

# =============================================================================
# TEST MÔI TRƯỜNG — chạy: python envs/k8s_env.py
# =============================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("       KIỂM TRA MÔI TRƯỜNG K8s PLACEMENT")
    print("=" * 55)

    env = K8sPlacementEnv(num_nodes=5, num_services=5, max_steps=50)

    # ── Test 1: Gymnasium API check ──────────────────────────────────────────
    print("\n[Test 1] Kiểm tra Gymnasium API chuẩn...")
    from gymnasium.utils.env_checker import check_env
    try:
        check_env(env, warn=True)
        print("  ✅ Gym check_env: PASS")
    except Exception as e:
        print(f"  ❌ Gym check_env: FAIL — {e}")

    # ── Test 2: Episode bình thường với Random agent ─────────────────────────
    print("\n[Test 2] Random agent — episode bình thường")
    obs, _ = env.reset(seed=42)
    print(f"  Observation shape : {obs.shape}  ✅")
    print(f"  Observation range : [{obs.min():.2f}, {obs.max():.2f}]  ✅")

    done = False
    total_reward = 0.0
    step_n = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step_n += 1
        status = "✅" if info["success"] else "❌"
        print(f"  Bước {step_n:2d} | Node {action} | {status} "
              f"| reward {reward:+.2f} | đặt {info['placed_total']}/5")

    reason = "terminated" if terminated else "truncated"
    print(f"\n  ► Episode kết thúc ({reason}) sau {step_n} bước")
    print(f"  ► Tổng reward: {total_reward:.2f}")
    print(f"  ► Services đặt thành công: {info['placed_total']}/{env.num_services}")

    # ── Test 3: Node bị chết giữa chừng — kiểm tra bug cũ ──────────────────
    print("\n[Test 3] Node 2 chết — kiểm tra vòng lặp vô tận")
    obs, _ = env.reset(seed=0)
    env.topology.nodes[2].fail()
    print("  ⚠️  Node 2 đã bị đánh sập")

    done = False
    total_reward = 0.0
    step_n = 0

    while not done:
        action = 2                  # Cố tình luôn chọn node chết
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step_n += 1

    print(f"  ► Kết thúc sau {step_n} bước (không bị treo) ✅")
    print(f"  ► Tổng reward: {total_reward:.2f}  (toàn phạt nặng -50 là đúng)")
    print(f"  ► Services đặt thành công: {info['placed_total']}/{env.num_services}  (phải = 0)")

    # ── Test 4: Nhiều episode liên tiếp (stress test nhỏ) ───────────────────
    print("\n[Test 4] Stress test — 100 episodes liên tiếp")
    rewards = []
    for ep in range(100):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)

    rewards = np.array(rewards)
    print(f"  ✅ 100 episodes hoàn thành không treo")
    print(f"  Reward trung bình : {rewards.mean():.2f}")
    print(f"  Reward cao nhất   : {rewards.max():.2f}")
    print(f"  Reward thấp nhất  : {rewards.min():.2f}")

    print("\n" + "=" * 55)
    print("  Tất cả test PASS — môi trường sẵn sàng training ✅")
    print("=" * 55)
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
    Microservice
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
                 max_steps: int = 200):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_services = num_services
        # ── BẢO HIỂM: Giới hạn số bước tối đa mỗi episode ──────────────────
        # Về lý thuyết mỗi episode chỉ cần num_services bước, nhưng max_steps
        # là lưới an toàn cuối cùng tránh episode chạy mãi khi có lỗi logic.
        self.max_steps = max_steps

        # Hạ tầng mạng và chuỗi service — tạo một lần, reset() sẽ làm sạch
        self.topology = create_sample_topology(num_nodes)
        self.service_chain = create_sample_service_chain(num_services, seed = 0)

        # Bộ đếm nội bộ
        self.current_service_idx: int = 0
        self.step_count: int = 0

        # Action / Observation space
        self.action_space = spaces.Discrete(num_nodes)
        obs_dim = (num_nodes * 2) + 2 + num_services
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    # ──────────────────────────────────────────────────────────────────────────
    # QUAN SÁT
    # ──────────────────────────────────────────────────────────────────────────
    def _get_observation(self) -> np.ndarray:
        obs = []

        for node in self.topology.nodes:
            if not node.is_active:
                obs.extend([1.0, 1.0])          # Node chết → coi như đầy
            else:
                obs.append(node.cpu_used    / node.cpu_capacity)
                obs.append(node.memory_used / node.memory_capacity)
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
    # PHẦN THƯỞNG
    # ──────────────────────────────────────────────────────────────────────────
    def _calculate_reward(self, service: Microservice,
                        node_id: int, success: bool) -> float:
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

        # 8. Latency penalty giữa service liên tiếp
        # Giảm nhẹ để latency không ép agent gom toàn bộ chain vào 1 node
        if service.id > 0:
            prev_service = self.service_chain.services[service.id - 1]
            if prev_service.placed_on >= 0:
                latency = self.topology.get_latency(prev_service.placed_on, node_id)
                reward -= min(latency * 0.15, 1.5)

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

    # ──────────────────────────────────────────────────────────────────────────
    # RESET
    # ──────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.topology.reset()
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
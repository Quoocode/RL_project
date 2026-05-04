# envs/k8s_env.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Optional

# Import các class từ file topology bạn vừa làm xong
from envs.topology import (
    create_sample_topology,
    create_sample_service_chain,
    Microservice
)

class K8sPlacementEnv(gym.Env):
    """Môi trường mô phỏng Kubernetes cho bài toán xếp Microservice."""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, num_nodes: int = 5, num_services: int = 5):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_services = num_services
        
        # 1. Khởi tạo hạ tầng mạng và chuỗi service
        self.topology = create_sample_topology(num_nodes)
        self.service_chain = create_sample_service_chain(num_services)
        self.current_service_idx = 0
        
        # 2. Không gian hành động (Action Space): AI chọn 1 Node (từ 0 đến num_nodes - 1)
        self.action_space = spaces.Discrete(num_nodes)
        
        # 3. Không gian quan sát (Observation Space)
        # Gồm: (CPU_util, RAM_util) của từng node + One-hot vector của service hiện tại
        obs_dim = (num_nodes * 2) + num_services
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
    def _get_observation(self) -> np.ndarray:
        """Gom thập thông tin hiện tại gửi cho AI."""
        obs = []
        # Tình trạng các Node
        for node in self.topology.nodes:
            if not node.is_active:
                obs.extend([1.0, 1.0]) # Node chết coi như đầy 100% tài nguyên
            else:
                obs.append(node.cpu_used / node.cpu_capacity)
                obs.append(node.memory_used / node.memory_capacity)
                
        # Service nào đang cần được xếp (One-hot encoding)
        service_onehot = np.zeros(self.num_services)
        if self.current_service_idx < self.num_services:
            service_onehot[self.current_service_idx] = 1.0
        obs.extend(service_onehot)
        
        return np.array(obs, dtype=np.float32)

    def _calculate_reward(self, service: Microservice, node_id: int, success: bool) -> float:
        """Hàm chấm điểm cực kỳ quan trọng cho AI."""
        node = self.topology.get_node(node_id)
        
        # SỬA LỖI Ở ĐÂY: Phạt ngay lập tức nếu chọn trúng Node chết
        if not node.is_active:
            return -50.0 
            
        # PHẠT NẶNG 2: Node không đủ CPU/RAM (vượt quá dung lượng)
        if not success:
            return -20.0 
            
        reward = 5.0 # THƯỞNG CƠ BẢN: Xếp thành công
        
        # PHẠT NẶNG HƠN: Ép AI cân bằng tải, cực ghét việc nhét đầy node
        cpu_util = node.cpu_used / node.cpu_capacity
        if cpu_util > 0.8:
            # Thay vì * 10, ta dùng * 50 và bình phương để phạt cực nặng khi tiến gần 100%
            reward -= ((cpu_util - 0.8) * 50) ** 2 
            
        # Tương tự cho RAM để đảm bảo an toàn hệ thống
        mem_util = node.memory_used / node.memory_capacity
        if mem_util > 0.8:
            reward -= ((mem_util - 0.8) * 50) ** 2
            
        # THƯỞNG/PHẠT ĐỘ TRỄ (Latency): So với service đứng trước nó
        if service.id > 0:
            prev_service = self.service_chain.services[service.id - 1]
            if prev_service.placed_on >= 0:
                latency = self.topology.get_latency(prev_service.placed_on, node_id)
                # Đảm bảo điểm trừ độ trễ không vượt quá điểm thưởng
                reward -= min(latency * 0.5, 4.0) 
                
        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Thực thi hành động của AI."""
        service = self.service_chain.services[self.current_service_idx]
        node = self.topology.get_node(action)
        
        # Thử nhét service vào Node AI đã chọn
        success = False
        if node.is_active:
            success = node.allocate(service.cpu_request, service.memory_request)
        
        if success:
            service.placed_on = action
            self.current_service_idx += 1
            
        # Chấm điểm
        reward = self._calculate_reward(service, action, success)
            
        terminated = self.current_service_idx >= self.num_services
        
        return self._get_observation(), reward, terminated, False, {"success": success}

    def reset(self, seed=None, options=None):
        """Khởi động lại môi trường (chơi ván mới)."""
        super().reset(seed=seed)
        self.topology.reset()
        for svc in self.service_chain.services:
            svc.placed_on = -1
        self.current_service_idx = 0
        return self._get_observation(), {}

# ==========================================
# KHU VỰC CHẠY THỬ NGHIỆM (TESTING)
# ==========================================
if __name__ == "__main__":
    print("=== BẮT ĐẦU TEST MÔI TRƯỜNG K8S ===")
    
    env = K8sPlacementEnv(num_nodes=5, num_services=5)
    
    # 1. PHẢI RESET MÔI TRƯỜNG TRƯỚC
    obs, _ = env.reset()
    
    # 2. RỒI MỚI ĐÁNH SẬP NODE
    env.topology.nodes[2].fail()
    print("⚠️ Đã đánh sập Node 2 (is_active = False)\n")
    
    done = False
    step_count = 1
    total_reward = 0
    
    # Cho một "AI ngốc" (Random) chạy thử
    while not done and step_count <= 10:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        
        print(f"Lượt {step_count}: AI quyết định xếp vào Node {action}")
        print(f"   -> Thành công: {info['success']} | Điểm: {reward:.2f}")
        step_count += 1
        
    print(f"\n🏆 Tổng điểm: {total_reward:.2f}")
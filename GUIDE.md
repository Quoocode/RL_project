# Hướng Dẫn Từng Bước - Đồ Án Microservice Chain Placement

## 📋 Mục Lục

1. [Giới Thiệu Dự Án](#1-giới-thiệu-dự-án)
2. [Kiến Thức Nền Tảng](#2-kiến-thức-nền-tảng)
3. [Cài Đặt Môi Trường](#3-cài-đặt-môi-trường)
4. [Cấu Trúc Dự Án](#4-cấu-trúc-dự-án)
5. [Bước 1: Xây Dựng Topology Mạng](#5-bước-1-xây-dựng-topology-mạng)
6. [Bước 2: Xây Dựng Kubernetes Environment](#6-bước-2-xây-dựng-kubernetes-environment)
7. [Bước 3: Xây Dựng Agent (DRL)](#7-bước-3-xây-dựng-agent-drl)
8. [Bước 4: Huấn Luyện Model](#8-bước-4-huấn-luyện-model)
9. [Bước 5: Đánh Giá Model](#9-bước-5-đánh-giá-model)
10. [Bước 6: Thực Nghiệm và Kết Quả](#10-bước-6-thực-nghiệm-và-kết-quả)
11. [Troubleshooting](#11-troubleshooting)
12. [Tài Liệu Tham Khảo](#12-tài-liệu-tham-khảo)

---

## 1. Giới Thiệu Dự Án

### 1.1 Mô Tả Bài Toán

Dự án này tập trung vào việc **tối ưu hóa việc sắp xếp chuỗi Microservice trong môi trường Kubernetes** sử dụng **Deep Reinforcement Learning (DRL)**.

**Bài toán đặt ra:**
- Trong hệ thống Kubernetes, có nhiều node (máy chủ) với tài nguyên giới hạn (CPU, RAM, Bandwidth)
- Cần triển khai các microservice thành chuỗi (Service Chain) lên các node
- Mục tiêu: Tối ưu hóa việc sắp xếp để giảm độ trễ, tăng throughput, và sử dụng tài nguyên hiệu quả

### 1.2 Kiến Trúc Giải Pháp

```
┌─────────────────────────────────────────────────────────────┐
│                    DRL Agent (PPO/A2C/DQN)                   │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│  │  State   │───▶│  Policy  │───▶│  Action  │               │
│  │ (Network │    │ Network  │    │(Placement│               │
│  │  State)  │    │          │    │ Decision)│               │
│  └──────────┘    └──────────┘    └──────────┘               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Kubernetes Environment                      │
│                                                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Node 1  │  │ Node 2  │  │ Node 3  │  │ Node N  │        │
│  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │        │
│  │ │ MS1 │ │  │ │ MS2 │ │  │ │ MS3 │ │  │ │ MSn │ │        │
│  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Kiến Thức Nền Tảng

### 2.1 Reinforcement Learning Cơ Bản

**Các khái niệm quan trọng:**

| Khái niệm | Mô tả |
|-----------|-------|
| **Agent** | Chương trình học cách ra quyết định |
| **Environment** | Môi trường mà agent tương tác |
| **State (s)** | Trạng thái hiện tại của môi trường |
| **Action (a)** | Hành động agent có thể thực hiện |
| **Reward (r)** | Phần thưởng nhận được sau mỗi action |
| **Policy (π)** | Chiến lược chọn action từ state |

### 2.2 Deep Reinforcement Learning

**Các thuật toán phổ biến:**

1. **DQN (Deep Q-Network)**: Sử dụng neural network để ước lượng Q-value
2. **PPO (Proximal Policy Optimization)**: Thuật toán policy gradient ổn định
3. **A2C (Advantage Actor-Critic)**: Kết hợp actor và critic network

### 2.3 Kubernetes Basics

**Các khái niệm cần nắm:**
- **Pod**: Đơn vị nhỏ nhất chứa container
- **Node**: Máy chủ vật lý/ảo chạy pod
- **Service**: Expose các pod ra bên ngoài
- **Deployment**: Quản lý việc triển khai pod

---

## 3. Cài Đặt Môi Trường

### 3.1 Yêu Cầu Hệ Thống

```
- Python 3.8+
- pip (Python package manager)
- Git
- (Optional) GPU với CUDA để tăng tốc training
```

### 3.2 Bước 1: Clone Repository

```bash
git clone https://github.com/PhamNTSang/Microservice-Chain-Placement.git
cd Microservice-Chain-Placement
```

### 3.3 Bước 2: Tạo Môi Trường Ảo

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt môi trường ảo
# Windows:
venv\Scripts\activate

# Linux/MacOS:
source venv/bin/activate
```

### 3.4 Bước 3: Cài Đặt Dependencies

Tạo file `requirements.txt`:

```txt
# Deep Learning Framework
torch>=2.0.0
torchvision>=0.15.0

# Reinforcement Learning
gymnasium>=0.28.0
stable-baselines3>=2.0.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.13.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0
```

Cài đặt:

```bash
pip install -r requirements.txt
```

### 3.5 Kiểm Tra Cài Đặt

```python
# test_setup.py
import torch
import gymnasium as gym
import stable_baselines3 as sb3
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Gymnasium version: {gym.__version__}")
print(f"Stable Baselines3 version: {sb3.__version__}")
print("✅ Setup successful!")
```

```bash
python test_setup.py
```

---

## 4. Cấu Trúc Dự Án

```
Microservice-Chain-Placement/
│
├── agents/                    # Các thuật toán DRL
│   ├── __init__.py
│   ├── dqn_agent.py          # DQN Agent
│   ├── ppo_agent.py          # PPO Agent
│   └── a2c_agent.py          # A2C Agent
│
├── envs/                      # Môi trường Kubernetes
│   ├── __init__.py
│   ├── k8s_env.py            # Kubernetes Environment
│   └── topology.py           # Network Topology
│
├── models/                    # Saved models
│   └── .gitkeep
│
├── logs/                      # Training logs
│   └── .gitkeep
│
├── configs/                   # Configuration files
│   └── default_config.yaml
│
├── main.py                    # Entry point
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── requirements.txt           # Dependencies
├── README.md                  # Project overview
└── GUIDE.md                   # This guide
```

---

## 5. Bước 1: Xây Dựng Topology Mạng

### 5.1 Mô Tả

File `envs/topology.py` định nghĩa cấu trúc mạng và các node trong cluster.

### 5.2 Code Implementation

```python
# envs/topology.py

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class Node:
    """Đại diện cho một node trong Kubernetes cluster."""
    id: int
    cpu_capacity: float      # Tổng CPU (cores)
    memory_capacity: float   # Tổng RAM (GB)
    bandwidth: float         # Bandwidth (Mbps)
    
    # Tài nguyên đang sử dụng
    cpu_used: float = 0.0
    memory_used: float = 0.0
    
    @property
    def cpu_available(self) -> float:
        return self.cpu_capacity - self.cpu_used
    
    @property
    def memory_available(self) -> float:
        return self.memory_capacity - self.memory_used
    
    def can_allocate(self, cpu: float, memory: float) -> bool:
        """Kiểm tra node có đủ tài nguyên không."""
        return self.cpu_available >= cpu and self.memory_available >= memory
    
    def allocate(self, cpu: float, memory: float) -> bool:
        """Cấp phát tài nguyên cho microservice."""
        if not self.can_allocate(cpu, memory):
            return False
        self.cpu_used += cpu
        self.memory_used += memory
        return True
    
    def deallocate(self, cpu: float, memory: float):
        """Giải phóng tài nguyên."""
        self.cpu_used = max(0, self.cpu_used - cpu)
        self.memory_used = max(0, self.memory_used - memory)
    
    def reset(self):
        """Reset node về trạng thái ban đầu."""
        self.cpu_used = 0.0
        self.memory_used = 0.0


@dataclass
class Microservice:
    """Đại diện cho một microservice."""
    id: int
    name: str
    cpu_request: float       # CPU yêu cầu (cores)
    memory_request: float    # RAM yêu cầu (GB)
    
    # Node hiện tại đang chứa microservice
    placed_on: int = -1


class ServiceChain:
    """Đại diện cho một chuỗi microservice."""
    
    def __init__(self, chain_id: int, services: List[Microservice]):
        self.chain_id = chain_id
        self.services = services
        self.latency_requirements = {}  # Yêu cầu về độ trễ
    
    def add_latency_requirement(self, src: int, dst: int, max_latency: float):
        """Thêm yêu cầu về độ trễ giữa 2 service."""
        self.latency_requirements[(src, dst)] = max_latency


class NetworkTopology:
    """Mô hình hóa topology mạng của cluster."""
    
    def __init__(self, num_nodes: int, config: Dict = None):
        self.num_nodes = num_nodes
        self.nodes: List[Node] = []
        self.latency_matrix = np.zeros((num_nodes, num_nodes))
        
        # Khởi tạo nodes với cấu hình mặc định hoặc custom
        self._initialize_nodes(config)
        self._initialize_latency_matrix()
    
    def _initialize_nodes(self, config: Dict = None):
        """Khởi tạo các nodes trong cluster."""
        default_config = {
            'cpu_capacity': 8,      # 8 cores
            'memory_capacity': 16,   # 16 GB
            'bandwidth': 1000       # 1 Gbps
        }
        
        if config is None:
            config = default_config
        
        for i in range(self.num_nodes):
            node = Node(
                id=i,
                cpu_capacity=config.get('cpu_capacity', 8),
                memory_capacity=config.get('memory_capacity', 16),
                bandwidth=config.get('bandwidth', 1000)
            )
            self.nodes.append(node)
    
    def _initialize_latency_matrix(self):
        """Khởi tạo ma trận độ trễ giữa các nodes."""
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    self.latency_matrix[i][j] = 0
                else:
                    # Giả lập độ trễ ngẫu nhiên (1-10 ms)
                    self.latency_matrix[i][j] = np.random.uniform(1, 10)
    
    def get_latency(self, src_node: int, dst_node: int) -> float:
        """Lấy độ trễ giữa 2 nodes."""
        return self.latency_matrix[src_node][dst_node]
    
    def get_node(self, node_id: int) -> Node:
        """Lấy thông tin node."""
        return self.nodes[node_id]
    
    def get_cluster_state(self) -> np.ndarray:
        """Trả về trạng thái hiện tại của cluster."""
        state = []
        for node in self.nodes:
            state.extend([
                node.cpu_used / node.cpu_capacity,      # CPU utilization
                node.memory_used / node.memory_capacity  # Memory utilization
            ])
        return np.array(state, dtype=np.float32)
    
    def reset(self):
        """Reset topology về trạng thái ban đầu."""
        for node in self.nodes:
            node.reset()


def create_sample_topology(num_nodes: int = 5) -> NetworkTopology:
    """Tạo một topology mẫu cho testing."""
    config = {
        'cpu_capacity': 8,
        'memory_capacity': 16,
        'bandwidth': 1000
    }
    return NetworkTopology(num_nodes, config)


def create_sample_service_chain() -> ServiceChain:
    """Tạo một service chain mẫu."""
    services = [
        Microservice(0, "api-gateway", 0.5, 1.0),
        Microservice(1, "auth-service", 0.3, 0.5),
        Microservice(2, "user-service", 0.5, 1.0),
        Microservice(3, "order-service", 0.8, 1.5),
        Microservice(4, "database", 1.0, 2.0),
    ]
    
    chain = ServiceChain(0, services)
    
    # Thêm yêu cầu về độ trễ
    chain.add_latency_requirement(0, 1, 5.0)  # api -> auth: max 5ms
    chain.add_latency_requirement(1, 2, 5.0)  # auth -> user: max 5ms
    chain.add_latency_requirement(2, 3, 10.0) # user -> order: max 10ms
    chain.add_latency_requirement(3, 4, 10.0) # order -> db: max 10ms
    
    return chain
```

### 5.3 Test Topology

```python
# Test topology
if __name__ == "__main__":
    # Tạo topology với 5 nodes
    topology = create_sample_topology(5)
    
    print("=== Cluster Information ===")
    for node in topology.nodes:
        print(f"Node {node.id}: CPU={node.cpu_capacity} cores, "
              f"RAM={node.memory_capacity} GB")
    
    print("\n=== Latency Matrix ===")
    print(topology.latency_matrix)
    
    # Test service chain
    chain = create_sample_service_chain()
    print(f"\n=== Service Chain {chain.chain_id} ===")
    for svc in chain.services:
        print(f"  {svc.name}: CPU={svc.cpu_request}, RAM={svc.memory_request}")
```

---

## 6. Bước 2: Xây Dựng Kubernetes Environment

### 6.1 Mô Tả

File `envs/k8s_env.py` định nghĩa môi trường RL theo chuẩn Gymnasium.

### 6.2 Code Implementation

```python
# envs/k8s_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional

from .topology import (
    NetworkTopology, 
    ServiceChain, 
    Microservice,
    create_sample_topology,
    create_sample_service_chain
)


class K8sPlacementEnv(gym.Env):
    """
    Kubernetes Service Placement Environment.
    
    Môi trường mô phỏng việc đặt các microservice lên các node
    trong Kubernetes cluster.
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        num_nodes: int = 5,
        num_services: int = 5,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_services = num_services
        self.render_mode = render_mode
        
        # Khởi tạo topology và service chain
        self.topology = create_sample_topology(num_nodes)
        self.service_chain = create_sample_service_chain()
        
        # Current service đang cần đặt
        self.current_service_idx = 0
        
        # Định nghĩa Action Space
        # Action = chọn node để đặt service hiện tại
        self.action_space = spaces.Discrete(num_nodes)
        
        # Định nghĩa Observation Space
        # State bao gồm:
        # - Trạng thái tài nguyên của từng node (cpu_util, mem_util) x num_nodes
        # - Service hiện tại cần đặt (one-hot encoding) x num_services
        # - Vị trí các service đã đặt (node_id / num_nodes) x num_services
        obs_dim = (num_nodes * 2) + num_services + num_services
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Tracking
        self.episode_reward = 0
        self.placement_history = []
    
    def _get_observation(self) -> np.ndarray:
        """Tạo observation từ trạng thái hiện tại."""
        obs = []
        
        # 1. Trạng thái tài nguyên các nodes
        for node in self.topology.nodes:
            obs.append(node.cpu_used / node.cpu_capacity)
            obs.append(node.memory_used / node.memory_capacity)
        
        # 2. One-hot encoding của service hiện tại
        current_service_onehot = np.zeros(self.num_services)
        if self.current_service_idx < self.num_services:
            current_service_onehot[self.current_service_idx] = 1.0
        obs.extend(current_service_onehot)
        
        # 3. Vị trí các service đã đặt (normalized)
        placement = np.zeros(self.num_services)
        for i, svc in enumerate(self.service_chain.services):
            if svc.placed_on >= 0:
                placement[i] = (svc.placed_on + 1) / self.num_nodes
        obs.extend(placement)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(
        self, 
        service: Microservice, 
        node_id: int,
        success: bool
    ) -> float:
        """Tính reward cho action."""
        if not success:
            return -10.0  # Penalty cho việc đặt thất bại
        
        reward = 0.0
        node = self.topology.get_node(node_id)
        
        # 1. Reward cho việc cân bằng tải
        cpu_util = node.cpu_used / node.cpu_capacity
        mem_util = node.memory_used / node.memory_capacity
        
        # Penalty nếu overload
        if cpu_util > 0.8:
            reward -= (cpu_util - 0.8) * 10
        if mem_util > 0.8:
            reward -= (mem_util - 0.8) * 10
        
        # 2. Reward cho việc giảm latency
        if service.id > 0:
            prev_service = self.service_chain.services[service.id - 1]
            if prev_service.placed_on >= 0:
                latency = self.topology.get_latency(
                    prev_service.placed_on, 
                    node_id
                )
                # Reward cao hơn nếu latency thấp
                reward += max(0, 10 - latency)
        
        # 3. Reward cơ bản cho việc đặt thành công
        reward += 5.0
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Thực hiện một action (đặt service lên node).
        
        Args:
            action: Node ID để đặt service hiện tại
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.current_service_idx >= self.num_services:
            # Đã đặt hết services
            return self._get_observation(), 0.0, True, False, {}
        
        # Lấy service hiện tại
        service = self.service_chain.services[self.current_service_idx]
        node = self.topology.get_node(action)
        
        # Thử đặt service lên node
        success = node.allocate(service.cpu_request, service.memory_request)
        
        if success:
            service.placed_on = action
            self.placement_history.append((service.id, action))
        
        # Tính reward
        reward = self._calculate_reward(service, action, success)
        self.episode_reward += reward
        
        # Chuyển sang service tiếp theo
        self.current_service_idx += 1
        
        # Kiểm tra kết thúc
        terminated = self.current_service_idx >= self.num_services
        
        info = {
            'service_id': service.id,
            'node_id': action,
            'success': success,
            'episode_reward': self.episode_reward
        }
        
        return self._get_observation(), reward, terminated, False, info
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset môi trường về trạng thái ban đầu."""
        super().reset(seed=seed)
        
        # Reset topology
        self.topology.reset()
        
        # Reset service chain
        for svc in self.service_chain.services:
            svc.placed_on = -1
        
        # Reset tracking
        self.current_service_idx = 0
        self.episode_reward = 0
        self.placement_history = []
        
        return self._get_observation(), {}
    
    def render(self):
        """Render trạng thái hiện tại."""
        if self.render_mode == 'human' or self.render_mode == 'ansi':
            print("\n" + "=" * 50)
            print("KUBERNETES CLUSTER STATUS")
            print("=" * 50)
            
            # In trạng thái nodes
            for node in self.topology.nodes:
                cpu_bar = "█" * int(10 * node.cpu_used / node.cpu_capacity)
                cpu_bar += "░" * (10 - len(cpu_bar))
                
                mem_bar = "█" * int(10 * node.memory_used / node.memory_capacity)
                mem_bar += "░" * (10 - len(mem_bar))
                
                services_on_node = [
                    svc.name for svc in self.service_chain.services 
                    if svc.placed_on == node.id
                ]
                
                print(f"\nNode {node.id}:")
                print(f"  CPU: [{cpu_bar}] {node.cpu_used:.1f}/{node.cpu_capacity}")
                print(f"  MEM: [{mem_bar}] {node.memory_used:.1f}/{node.memory_capacity}")
                print(f"  Services: {services_on_node}")
            
            print("\n" + "=" * 50)
    
    def close(self):
        """Cleanup resources."""
        pass


# Register environment với Gymnasium
gym.register(
    id='K8sPlacement-v0',
    entry_point='envs.k8s_env:K8sPlacementEnv',
)
```

### 6.3 Test Environment

```python
# Test environment
if __name__ == "__main__":
    import gymnasium as gym
    
    # Tạo environment
    env = K8sPlacementEnv(num_nodes=5, num_services=5, render_mode='human')
    
    # Reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Random actions
    total_reward = 0
    done = False
    
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Action: {action}, Reward: {reward:.2f}, "
              f"Success: {info.get('success', True)}")
        
        done = terminated or truncated
    
    env.render()
    print(f"\nTotal Reward: {total_reward:.2f}")
    env.close()
```

---

## 7. Bước 3: Xây Dựng Agent (DRL)

### 7.1 Sử Dụng Stable Baselines3

Stable Baselines3 cung cấp các thuật toán DRL đã được implement sẵn.

### 7.2 PPO Agent

```python
# agents/ppo_agent.py

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
import numpy as np
from typing import Optional


class PlacementCallback(BaseCallback):
    """Custom callback để monitor training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log rewards
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(ep_info['r'])
            self.episode_lengths.append(ep_info['l'])
            
            if self.verbose > 0 and len(self.episode_rewards) % 100 == 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {len(self.episode_rewards)}, "
                      f"Mean Reward (last 100): {mean_reward:.2f}")
        
        return True


class PPOAgent:
    """PPO Agent cho bài toán Service Placement."""
    
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        device: str = 'auto'
    ):
        self.env = env
        self.device = device
        
        # Khởi tạo PPO model
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            verbose=1,
            device=device,
            tensorboard_log="./logs/"
        )
    
    def train(
        self, 
        total_timesteps: int = 100000,
        callback: Optional[BaseCallback] = None
    ):
        """Train agent."""
        if callback is None:
            callback = PlacementCallback(verbose=1)
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    
    def predict(self, observation, deterministic: bool = True):
        """Dự đoán action từ observation."""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def save(self, path: str):
        """Lưu model."""
        self.model.save(path)
    
    def load(self, path: str):
        """Load model."""
        self.model = PPO.load(path, env=self.env)
    
    def evaluate(self, n_episodes: int = 10) -> dict:
        """Đánh giá agent."""
        rewards = []
        successes = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_success = True
            
            while not done:
                action = self.predict(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                
                if not info.get('success', True):
                    episode_success = False
                
                done = terminated or truncated
            
            rewards.append(episode_reward)
            successes.append(episode_success)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'success_rate': np.mean(successes)
        }
```

### 7.3 DQN Agent

```python
# agents/dqn_agent.py

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from typing import Optional


class DQNAgent:
    """DQN Agent cho bài toán Service Placement."""
    
    def __init__(
        self,
        env,
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        batch_size: int = 32,
        gamma: float = 0.99,
        exploration_fraction: float = 0.1,
        exploration_final_eps: float = 0.05,
        device: str = 'auto'
    ):
        self.env = env
        self.device = device
        
        self.model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            verbose=1,
            device=device,
            tensorboard_log="./logs/"
        )
    
    def train(self, total_timesteps: int = 100000, callback=None):
        """Train agent."""
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    
    def predict(self, observation, deterministic: bool = True):
        """Dự đoán action."""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def save(self, path: str):
        """Lưu model."""
        self.model.save(path)
    
    def load(self, path: str):
        """Load model."""
        self.model = DQN.load(path, env=self.env)
    
    def evaluate(self, n_episodes: int = 10) -> dict:
        """Đánh giá agent."""
        rewards = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.predict(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards)
        }
```

### 7.4 A2C Agent

```python
# agents/a2c_agent.py

from stable_baselines3 import A2C
import numpy as np
from typing import Optional


class A2CAgent:
    """A2C Agent cho bài toán Service Placement."""
    
    def __init__(
        self,
        env,
        learning_rate: float = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        device: str = 'auto'
    ):
        self.env = env
        self.device = device
        
        self.model = A2C(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            verbose=1,
            device=device,
            tensorboard_log="./logs/"
        )
    
    def train(self, total_timesteps: int = 100000, callback=None):
        """Train agent."""
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    
    def predict(self, observation, deterministic: bool = True):
        """Dự đoán action."""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def save(self, path: str):
        """Lưu model."""
        self.model.save(path)
    
    def load(self, path: str):
        """Load model."""
        self.model = A2C.load(path, env=self.env)
```

---

## 8. Bước 4: Huấn Luyện Model

### 8.1 Training Script

```python
# train.py

import argparse
import os
from datetime import datetime

from envs.k8s_env import K8sPlacementEnv
from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent
from agents.a2c_agent import A2CAgent


def parse_args():
    parser = argparse.ArgumentParser(description='Train DRL Agent')
    
    parser.add_argument('--agent', type=str, default='ppo',
                        choices=['ppo', 'dqn', 'a2c'],
                        help='Agent type')
    parser.add_argument('--num_nodes', type=int, default=5,
                        help='Number of nodes in cluster')
    parser.add_argument('--num_services', type=int, default=5,
                        help='Number of services in chain')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--save_path', type=str, default='./models',
                        help='Path to save trained model')
    
    return parser.parse_args()


def create_agent(agent_type: str, env, learning_rate: float):
    """Factory function để tạo agent."""
    agents = {
        'ppo': PPOAgent,
        'dqn': DQNAgent,
        'a2c': A2CAgent
    }
    
    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agents[agent_type](env, learning_rate=learning_rate)


def main():
    args = parse_args()
    
    # Tạo thư mục lưu model
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # Tạo environment
    print(f"Creating environment with {args.num_nodes} nodes "
          f"and {args.num_services} services...")
    env = K8sPlacementEnv(
        num_nodes=args.num_nodes,
        num_services=args.num_services
    )
    
    # Tạo agent
    print(f"Creating {args.agent.upper()} agent...")
    agent = create_agent(args.agent, env, args.lr)
    
    # Training
    print(f"Starting training for {args.timesteps} timesteps...")
    agent.train(total_timesteps=args.timesteps)
    
    # Lưu model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        args.save_path, 
        f"{args.agent}_{args.num_nodes}nodes_{timestamp}"
    )
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Đánh giá cuối cùng
    print("\n=== Final Evaluation ===")
    results = agent.evaluate(n_episodes=20)
    print(f"Mean Reward: {results['mean_reward']:.2f} "
          f"(± {results['std_reward']:.2f})")
    
    env.close()


if __name__ == "__main__":
    main()
```

### 8.2 Chạy Training

```bash
# Train với PPO (default)
python train.py --timesteps 50000

# Train với DQN
python train.py --agent dqn --timesteps 100000

# Train với nhiều nodes hơn
python train.py --num_nodes 10 --timesteps 200000

# Xem logs với Tensorboard
tensorboard --logdir ./logs
```

---

## 9. Bước 5: Đánh Giá Model

### 9.1 Evaluation Script

```python
# evaluate.py

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from envs.k8s_env import K8sPlacementEnv
from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent
from agents.a2c_agent import A2CAgent


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DRL Agent')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--agent', type=str, default='ppo',
                        choices=['ppo', 'dqn', 'a2c'],
                        help='Agent type')
    parser.add_argument('--num_nodes', type=int, default=5,
                        help='Number of nodes')
    parser.add_argument('--num_services', type=int, default=5,
                        help='Number of services')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render environment')
    
    return parser.parse_args()


def evaluate_agent(agent, env, n_episodes: int, render: bool = False):
    """Đánh giá chi tiết agent."""
    results = {
        'rewards': [],
        'placement_success': [],
        'latencies': [],
        'resource_utilization': []
    }
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        all_success = True
        
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if not info.get('success', True):
                all_success = False
            
            done = terminated or truncated
        
        # Tính resource utilization
        total_cpu_util = 0
        total_mem_util = 0
        for node in env.topology.nodes:
            total_cpu_util += node.cpu_used / node.cpu_capacity
            total_mem_util += node.memory_used / node.memory_capacity
        avg_util = (total_cpu_util + total_mem_util) / (2 * env.num_nodes)
        
        results['rewards'].append(episode_reward)
        results['placement_success'].append(all_success)
        results['resource_utilization'].append(avg_util)
        
        if render and ep < 3:  # Render first 3 episodes
            env.render()
    
    return results


def plot_results(results: dict, save_path: str = None):
    """Vẽ biểu đồ kết quả."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Reward distribution
    axes[0, 0].hist(results['rewards'], bins=20, edgecolor='black')
    axes[0, 0].set_xlabel('Episode Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Reward Distribution')
    axes[0, 0].axvline(np.mean(results['rewards']), color='r', 
                       linestyle='--', label=f'Mean: {np.mean(results["rewards"]):.2f}')
    axes[0, 0].legend()
    
    # Reward over episodes
    axes[0, 1].plot(results['rewards'])
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].set_title('Reward over Episodes')
    
    # Success rate
    success_rate = np.mean(results['placement_success']) * 100
    axes[1, 0].bar(['Success Rate'], [success_rate])
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].set_ylabel('Percentage')
    axes[1, 0].set_title(f'Placement Success Rate: {success_rate:.1f}%')
    
    # Resource utilization
    axes[1, 1].hist(results['resource_utilization'], bins=20, edgecolor='black')
    axes[1, 1].set_xlabel('Average Resource Utilization')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Resource Utilization Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()


def compare_with_baselines(env, agent, n_episodes: int = 50):
    """So sánh với các baseline algorithms."""
    
    # 1. Random placement
    random_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward
            done = terminated or truncated
        random_rewards.append(reward_sum)
    
    # 2. First-Fit placement
    ff_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        reward_sum = 0
        while not done:
            # Chọn node đầu tiên có đủ tài nguyên
            action = 0
            for i, node in enumerate(env.topology.nodes):
                svc = env.service_chain.services[env.current_service_idx]
                if node.can_allocate(svc.cpu_request, svc.memory_request):
                    action = i
                    break
            obs, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward
            done = terminated or truncated
        ff_rewards.append(reward_sum)
    
    # 3. DRL agent
    drl_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward
            done = terminated or truncated
        drl_rewards.append(reward_sum)
    
    # Vẽ so sánh
    methods = ['Random', 'First-Fit', 'DRL']
    means = [np.mean(random_rewards), np.mean(ff_rewards), np.mean(drl_rewards)]
    stds = [np.std(random_rewards), np.std(ff_rewards), np.std(drl_rewards)]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, means, yerr=stds, capsize=5)
    plt.ylabel('Average Reward')
    plt.title('Comparison with Baseline Methods')
    
    # Highlight best
    best_idx = np.argmax(means)
    bars[best_idx].set_color('green')
    
    plt.tight_layout()
    plt.savefig('./results/comparison.png')
    plt.show()
    
    return {
        'random': {'mean': np.mean(random_rewards), 'std': np.std(random_rewards)},
        'first_fit': {'mean': np.mean(ff_rewards), 'std': np.std(ff_rewards)},
        'drl': {'mean': np.mean(drl_rewards), 'std': np.std(drl_rewards)}
    }


def main():
    args = parse_args()
    
    # Tạo environment
    render_mode = 'human' if args.render else None
    env = K8sPlacementEnv(
        num_nodes=args.num_nodes,
        num_services=args.num_services,
        render_mode=render_mode
    )
    
    # Load agent
    agent_classes = {'ppo': PPOAgent, 'dqn': DQNAgent, 'a2c': A2CAgent}
    agent = agent_classes[args.agent](env)
    agent.load(args.model_path)
    
    print(f"Loaded {args.agent.upper()} agent from {args.model_path}")
    
    # Đánh giá
    print(f"\nEvaluating for {args.episodes} episodes...")
    results = evaluate_agent(agent, env, args.episodes, args.render)
    
    # In kết quả
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean Reward: {np.mean(results['rewards']):.2f} "
          f"(± {np.std(results['rewards']):.2f})")
    print(f"Success Rate: {np.mean(results['placement_success']) * 100:.1f}%")
    print(f"Avg Resource Utilization: {np.mean(results['resource_utilization']):.2%}")
    
    # Vẽ biểu đồ
    os.makedirs('./results', exist_ok=True)
    plot_results(results, './results/evaluation.png')
    
    # So sánh với baselines
    print("\n=== Comparing with Baselines ===")
    comparison = compare_with_baselines(env, agent)
    
    for method, stats in comparison.items():
        print(f"{method}: {stats['mean']:.2f} (± {stats['std']:.2f})")
    
    env.close()


if __name__ == "__main__":
    main()
```

### 9.2 Chạy Evaluation

```bash
# Đánh giá model
python evaluate.py --model_path ./models/ppo_5nodes_20240101_120000 --episodes 100

# Đánh giá với render
python evaluate.py --model_path ./models/ppo_5nodes_20240101_120000 --render
```

---

## 10. Bước 6: Thực Nghiệm và Kết Quả

### 10.1 Main Entry Point

```python
# main.py

import argparse
from train import main as train_main
from evaluate import main as evaluate_main


def main():
    parser = argparse.ArgumentParser(
        description='Microservice Chain Placement using DRL'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train agent')
    train_parser.add_argument('--agent', type=str, default='ppo',
                              choices=['ppo', 'dqn', 'a2c'])
    train_parser.add_argument('--timesteps', type=int, default=100000)
    train_parser.add_argument('--num_nodes', type=int, default=5)
    train_parser.add_argument('--num_services', type=int, default=5)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate agent')
    eval_parser.add_argument('--model_path', type=str, required=True)
    eval_parser.add_argument('--agent', type=str, default='ppo')
    eval_parser.add_argument('--episodes', type=int, default=100)
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("Starting training...")
        # Call training
    elif args.command == 'evaluate':
        print("Starting evaluation...")
        # Call evaluation
    elif args.command == 'demo':
        run_demo()
    else:
        parser.print_help()


def run_demo():
    """Demo nhanh để test hệ thống."""
    from envs.k8s_env import K8sPlacementEnv
    
    print("=" * 50)
    print("MICROSERVICE CHAIN PLACEMENT DEMO")
    print("=" * 50)
    
    env = K8sPlacementEnv(num_nodes=5, num_services=5, render_mode='human')
    obs, _ = env.reset()
    
    print("\nInitial State:")
    env.render()
    
    print("\nPlacing services with random policy...")
    done = False
    step = 0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        svc = env.service_chain.services[step]
        print(f"\nStep {step + 1}: Place '{svc.name}' on Node {action}")
        print(f"  Reward: {reward:.2f}, Success: {info.get('success', True)}")
        
        step += 1
        done = terminated or truncated
    
    print("\n" + "=" * 50)
    print("FINAL STATE")
    print("=" * 50)
    env.render()
    
    env.close()


if __name__ == "__main__":
    main()
```

### 10.2 Kế Hoạch Thực Nghiệm

| Thực nghiệm | Mô tả | Mục tiêu |
|-------------|-------|----------|
| **EXP-1** | So sánh PPO vs DQN vs A2C | Tìm thuật toán tốt nhất |
| **EXP-2** | Thay đổi số lượng nodes | Đánh giá scalability |
| **EXP-3** | Thay đổi số lượng services | Đánh giá với chuỗi dài hơn |
| **EXP-4** | So sánh với baselines | Đánh giá hiệu quả của DRL |
| **EXP-5** | Tài nguyên không đồng nhất | Đánh giá với heterogeneous cluster |

### 10.3 Cấu Trúc Báo Cáo

```
1. Giới thiệu
   - Bối cảnh và động lực
   - Mục tiêu nghiên cứu
   - Đóng góp chính

2. Kiến thức nền tảng
   - Kubernetes và Microservices
   - Reinforcement Learning
   - Các công trình liên quan

3. Phương pháp đề xuất
   - Mô hình hóa bài toán
   - Thiết kế môi trường
   - Thiết kế agent

4. Thực nghiệm
   - Thiết lập thực nghiệm
   - Kết quả và phân tích
   - So sánh với baselines

5. Kết luận
   - Tóm tắt kết quả
   - Hạn chế
   - Hướng phát triển
```

---

## 11. Troubleshooting

### 11.1 Lỗi Thường Gặp

#### Lỗi: "No module named 'envs'"

```bash
# Thêm thư mục gốc vào PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Hoặc trong code
import sys
sys.path.append('.')
```

#### Lỗi: "CUDA out of memory"

```python
# Giảm batch size
agent = PPOAgent(env, batch_size=32)

# Hoặc sử dụng CPU
agent = PPOAgent(env, device='cpu')
```

#### Lỗi: "Environment not registered"

```python
# Đảm bảo import trước khi sử dụng
from envs.k8s_env import K8sPlacementEnv

# Hoặc tạo trực tiếp
env = K8sPlacementEnv()
```

### 11.2 Tips Tối Ưu

1. **Training không hội tụ:**
   - Tăng số timesteps
   - Điều chỉnh learning rate
   - Kiểm tra reward function

2. **Training quá chậm:**
   - Sử dụng GPU
   - Giảm observation space
   - Vectorize environment

3. **Model overfit:**
   - Thêm noise vào environment
   - Sử dụng entropy regularization
   - Data augmentation

---

## 12. Tài Liệu Tham Khảo

### Sách và Papers

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
2. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*
3. Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*

### Documentation

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

### Online Courses

- [Deep RL Course by HuggingFace](https://huggingface.co/learn/deep-rl-course)
- [Spinning Up by OpenAI](https://spinningup.openai.com/)

---

## 📝 Checklist Hoàn Thành

- [ ] Cài đặt môi trường
- [ ] Implement `topology.py`
- [ ] Implement `k8s_env.py`
- [ ] Implement agents (PPO/DQN/A2C)
- [ ] Viết `train.py`
- [ ] Viết `evaluate.py`
- [ ] Training và lưu model
- [ ] Đánh giá và so sánh
- [ ] Viết báo cáo

---

**Chúc bạn hoàn thành đồ án thành công! 🎉**

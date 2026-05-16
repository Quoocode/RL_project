# envs/topology.py

import numpy as np
from dataclasses import dataclass
from typing import List, Dict

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
    
    # Trạng thái của Node (Dùng cho kịch bản Node Failure)
    is_active: bool = True
    
    @property
    def cpu_available(self) -> float:
        """Tính toán CPU còn trống. Nếu node chết thì trả về 0."""
        if not self.is_active: return 0.0
        return self.cpu_capacity - self.cpu_used
    
    @property
    def memory_available(self) -> float:
        """Tính toán RAM còn trống. Nếu node chết thì trả về 0."""
        if not self.is_active: return 0.0
        return self.memory_capacity - self.memory_used
    
    def can_allocate(self, cpu: float, memory: float) -> bool:
        """Kiểm tra node có đủ tài nguyên và đang hoạt động không."""
        if not self.is_active: return False
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
    
    def fail(self):
        """Mô phỏng Node bị sập đột ngột (Node Failure)."""
        self.is_active = False
        self.cpu_used = 0.0
        self.memory_used = 0.0
        
    def recover(self):
        """Mô phỏng Node được khôi phục."""
        self.is_active = True
        self.cpu_used = 0.0
        self.memory_used = 0.0

    def reset(self):
        """Reset node về trạng thái ban đầu."""
        self.is_active = True
        self.cpu_used = 0.0
        self.memory_used = 0.0


@dataclass
class Microservice:
    """Đại diện cho một microservice."""
    id: int
    name: str
    cpu_request: float       # CPU yêu cầu (cores)
    memory_request: float    # RAM yêu cầu (GB)
    
    # Node hiện tại đang chứa microservice (-1 nghĩa là chưa được đặt)
    placed_on: int = -1


class ServiceChain:
    """Đại diện cho một chuỗi microservice."""
    def __init__(self, chain_id: int, services: List[Microservice]):
        self.chain_id = chain_id
        self.services = services
        self.latency_requirements = {}  # Yêu cầu về độ trễ
    
    def add_latency_requirement(self, src: int, dst: int, max_latency: float):
        """Thêm yêu cầu về độ trễ tối đa giữa 2 service."""
        self.latency_requirements[(src, dst)] = max_latency


class NetworkTopology:
    """Mô hình hóa topology mạng của cluster."""
    def __init__(self, num_nodes: int, config: Dict = None):
        self.num_nodes = num_nodes
        self.nodes: List[Node] = []
        self.latency_matrix = np.zeros((num_nodes, num_nodes))
        
        self._initialize_nodes(config)
        self._initialize_latency_matrix()

    def _initialize_nodes(self, config: Dict = None):
        """
        Khởi tạo các node trong cluster.

        Env v2: sử dụng heterogeneous nodes thay vì tất cả node giống nhau.
        Với 5 node mặc định:
            Node 0: small   - 2 CPU / 4 GB  / 500 Mbps
            Node 1: medium  - 4 CPU / 8 GB  / 1000 Mbps
            Node 2: large   - 8 CPU / 16 GB / 2000 Mbps
            Node 3: medium  - 4 CPU / 8 GB  / 1000 Mbps
            Node 4: small   - 2 CPU / 4 GB  / 500 Mbps

        Nếu num_nodes khác 5, pattern này sẽ được lặp lại.
        """
        node_profiles = [
            {"name": "small",  "cpu": 2.0, "memory": 4.0,  "bandwidth": 500},
            {"name": "medium", "cpu": 4.0, "memory": 8.0,  "bandwidth": 1000},
            {"name": "large",  "cpu": 8.0, "memory": 16.0, "bandwidth": 2000},
            {"name": "medium", "cpu": 4.0, "memory": 8.0,  "bandwidth": 1000},
            {"name": "small",  "cpu": 2.0, "memory": 4.0,  "bandwidth": 500},
        ]

        for i in range(self.num_nodes):
            profile = node_profiles[i % len(node_profiles)]

            self.nodes.append(Node(
                id=i,
                cpu_capacity=profile["cpu"],
                memory_capacity=profile["memory"],
                bandwidth=profile["bandwidth"],
            ))
    
    def _initialize_latency_matrix(self):
        """Khởi tạo ma trận độ trễ ngẫu nhiên giữa các nodes (từ 1-10 ms)."""
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    self.latency_matrix[i][j] = 0
                else:
                    self.latency_matrix[i][j] = np.random.uniform(1, 10)
    
    def get_latency(self, src_node: int, dst_node: int) -> float:
        """Lấy độ trễ giữa 2 nodes. Trả về vô cực nếu 1 trong 2 node bị sập."""
        if not self.nodes[src_node].is_active or not self.nodes[dst_node].is_active:
            return float('inf') 
        return self.latency_matrix[src_node][dst_node]
    
    def get_node(self, node_id: int) -> Node:
        return self.nodes[node_id]
    
    def reset(self):
        """Reset toàn bộ topology."""
        for node in self.nodes:
            node.reset()

def create_sample_topology(num_nodes: int = 5) -> NetworkTopology:
    """
    Tạo topology mẫu cho môi trường mô phỏng.

    Env v2 mặc định dùng heterogeneous nodes.
    """
    return NetworkTopology(num_nodes)

def create_sample_service_chain(num_services: int = 5, seed=None) -> ServiceChain:
    if seed is not None:
        np.random.seed(seed)
    services = []
    for i in range(num_services):
        cpu_req = round(np.random.uniform(0.5, 2.0), 2)
        mem_req = round(np.random.uniform(1.0, 4.0), 2)
        services.append(Microservice(i, f"service-{i}", cpu_req, mem_req))
    chain = ServiceChain(0, services)
    for i in range(num_services - 1):
        chain.add_latency_requirement(i, i + 1, 5.0)
    return chain

if __name__ == "__main__":
    # Test nhanh xem file chạy ổn không
    print("=== Khởi tạo Topology với 3 Node ===")
    topology = create_sample_topology(3)
    for node in topology.nodes:
        print(f"Node {node.id}: Active={node.is_active}, CPU={node.cpu_capacity}, RAM={node.memory_capacity}")
    
    print("\n=== Kịch bản: Giả lập Node 1 bị sập đột ngột ===")
    topology.nodes[1].fail()
    print(f"Node 1 Active: {topology.nodes[1].is_active}")
    print(f"CPU rảnh của Node 1: {topology.nodes[1].cpu_available} (Dù tổng capacity là {topology.nodes[1].cpu_capacity})")
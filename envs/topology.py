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
        """Khởi tạo các nodes trong cluster."""
        default_config = {
            'cpu_capacity': 8,      # 8 cores
            'memory_capacity': 16,  # 16 GB
            'bandwidth': 1000       # 1 Gbps
        }
        config = config or default_config
        
        for i in range(self.num_nodes):
            node = Node(
                id=i,
                cpu_capacity=config.get('cpu_capacity', 8),
                memory_capacity=config.get('memory_capacity', 16),
                bandwidth=config.get('bandwidth', 1000)
            )
            self.nodes.append(node)
    
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
    config = {'cpu_capacity': 8, 'memory_capacity': 16, 'bandwidth': 1000}
    return NetworkTopology(num_nodes, config)

def create_sample_service_chain(num_services: int = 5) -> ServiceChain:
    np.random.seed(42) 
    services = []
    
    for i in range(num_services):
        # Tạo ngẫu nhiên CPU (0.1 - 0.8) và RAM (0.5 - 1.5) cho từng service
        # Nhỏ gọn để 30 cái vẫn có cơ hội nhét vừa 5 Node
        cpu_req = round(np.random.uniform(0.1, 0.8), 2)
        mem_req = round(np.random.uniform(0.5, 1.5), 2)
        
        services.append(Microservice(i, f"service-{i}", cpu_req, mem_req))
    
    chain = ServiceChain(0, services)
    
    # Thêm yêu cầu độ trễ (latency) cho các service nối tiếp nhau
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
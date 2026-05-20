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
        """Reset node về trạng thái ban đầu.

        Nếu node là padding (capacity == 0.0) thì giữ `is_active=False`.
        """
        self.cpu_used = 0.0
        self.memory_used = 0.0
        if self.cpu_capacity == 0.0 and self.memory_capacity == 0.0:
            # padding node — keep inactive
            self.is_active = False
        else:
            self.is_active = True


@dataclass
class Microservice:
    """Đại diện cho một microservice."""
    id: int
    name: str
    cpu_request: float       # CPU yêu cầu (cores)
    memory_request: float    # RAM yêu cầu (GB)

    # Loại workload: tiny, cpu-heavy, memory-heavy, balanced, large
    service_type: str = "balanced"
    
    # Node hiện tại đang chứa microservice (-1 nghĩa là chưa được đặt)
    placed_on: int = -1

@dataclass
class DependencyEdge:
    """
    Cạnh phụ thuộc giữa hai microservice.

    traffic_weight:
        Mức độ gọi / lưu lượng giữa hai service.
        Weight càng cao thì latency giữa hai service càng quan trọng.
    """
    src: int
    dst: int
    max_latency: float = 5.0
    traffic_weight: float = 1.0

class ServiceChain:
    """Đại diện cho một chuỗi / graph microservice."""
    def __init__(self, chain_id: int, services: List[Microservice]):
        self.chain_id = chain_id
        self.services = services

        # Backward compatibility:
        # Các phần code cũ vẫn có thể dùng latency_requirements[(src, dst)]
        self.latency_requirements = {}

        # Env v3:
        # Dependency graph có trọng số traffic.
        self.dependency_edges: List[DependencyEdge] = []

    def add_dependency(self,
                       src: int,
                       dst: int,
                       max_latency: float = 5.0,
                       traffic_weight: float = 1.0):
        """
        Thêm dependency edge giữa hai service.

        src, dst:
            ID của source/destination service.

        max_latency:
            Ngưỡng latency mong muốn.

        traffic_weight:
            Mức độ quan trọng / lưu lượng giữa hai service.
        """
        if src < 0 or src >= len(self.services):
            raise ValueError(f"Invalid src service id: {src}")

        if dst < 0 or dst >= len(self.services):
            raise ValueError(f"Invalid dst service id: {dst}")

        if src == dst:
            raise ValueError("Dependency edge cannot connect a service to itself")

        edge = DependencyEdge(
            src=src,
            dst=dst,
            max_latency=float(max_latency),
            traffic_weight=float(traffic_weight),
        )

        self.dependency_edges.append(edge)

        # Giữ lại dictionary cũ để không phá code hiện tại.
        self.latency_requirements[(src, dst)] = float(max_latency)

    def add_latency_requirement(self, src: int, dst: int, max_latency: float):
        """
        Hàm cũ, giữ lại để backward compatibility.

        Mặc định traffic_weight = 1.0.
        """
        self.add_dependency(
            src=src,
            dst=dst,
            max_latency=max_latency,
            traffic_weight=1.0,
        )

    def get_dependencies(self) -> List[DependencyEdge]:
        """Trả về danh sách dependency edges."""
        return self.dependency_edges

SERVICE_PROFILES = {
    # Service rất nhẹ, dùng để mô phỏng sidecar/helper service
    "tiny": {
        "cpu": (0.10, 0.35),
        "memory": (0.20, 0.80),
    },

    # CPU-heavy: phù hợp với xử lý tính toán
    "cpu-heavy": {
        "cpu": (1.30, 2.00),
        "memory": (0.60, 1.60),
    },

    # Memory-heavy: phù hợp với cache, in-memory processing
    "memory-heavy": {
        "cpu": (0.30, 0.90),
        "memory": (2.50, 4.00),
    },

    # Balanced: CPU/RAM tương đối vừa
    "balanced": {
        "cpu": (0.70, 1.30),
        "memory": (1.20, 2.60),
    },

    # Large: gần sát giới hạn của small node
    "large": {
        "cpu": (1.50, 2.00),
        "memory": (3.00, 4.00),
    },
}

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

def create_sample_topology(actual_num_nodes: int = 5, max_nodes: int = None) -> NetworkTopology:
    """
    Tạo topology mẫu cho môi trường mô phỏng.

    - `actual_num_nodes`: số node thực tế trong scenario (ví dụ 5).
    - `max_nodes`: nếu cung cấp và lớn hơn `actual_num_nodes`, hàm sẽ
      tạo topology với `max_nodes` nodes nhưng đánh dấu các node có id
      >= actual_num_nodes là inactive và đặt capacity = 0.0 để chúng
      đóng vai trò padding.

    Trả về `NetworkTopology` có số node bằng `max_nodes` nếu được chỉ
    định, hoặc `actual_num_nodes` nếu không.
    """

    total = max_nodes if (max_nodes is not None and max_nodes > actual_num_nodes) else actual_num_nodes
    topo = NetworkTopology(total)

    # Mark padding nodes (if any) as inactive and zero-capacity so they
    # won't affect normalization calculations.
    if total > actual_num_nodes:
        for i in range(actual_num_nodes, total):
            node = topo.nodes[i]
            node.is_active = False
            node.cpu_capacity = 0.0
            node.memory_capacity = 0.0
            node.cpu_used = 0.0
            node.memory_used = 0.0

    return topo

def create_sample_service_chain(num_services: int = 5, seed=None) -> ServiceChain:
    """
    Tạo chuỗi microservice với workload đa dạng.

    Env v2:
    - Không còn sinh CPU/RAM đều đều trong một khoảng chung.
    - Mỗi service thuộc một profile cụ thể:
        tiny, cpu-heavy, memory-heavy, balanced, large
    - Với num_services=5, mỗi episode sẽ có đủ 5 loại service,
      nhưng thứ tự được shuffle theo seed.
    """
    rng = np.random.default_rng(seed)

    profile_names = list(SERVICE_PROFILES.keys())

    # Nếu số service <= số profile, lấy một tập profile rồi shuffle.
    # Với 5 services mặc định, mỗi episode có đủ:
    # tiny, cpu-heavy, memory-heavy, balanced, large.
    if num_services <= len(profile_names):
        selected_profiles = profile_names.copy()
        rng.shuffle(selected_profiles)
        selected_profiles = selected_profiles[:num_services]
    else:
        # Nếu service nhiều hơn profile, sample thêm theo xác suất.
        # Có chủ ý cho balanced/cpu-heavy/memory-heavy xuất hiện nhiều hơn.
        weights = np.array([
            0.15,  # tiny
            0.25,  # cpu-heavy
            0.25,  # memory-heavy
            0.25,  # balanced
            0.10,  # large
        ], dtype=float)
        weights = weights / weights.sum()

        selected_profiles = list(rng.choice(
            profile_names,
            size=num_services,
            replace=True,
            p=weights
        ))

    services = []

    for i, service_type in enumerate(selected_profiles):
        profile = SERVICE_PROFILES[service_type]

        cpu_low, cpu_high = profile["cpu"]
        mem_low, mem_high = profile["memory"]

        cpu_req = round(float(rng.uniform(cpu_low, cpu_high)), 2)
        mem_req = round(float(rng.uniform(mem_low, mem_high)), 2)

        services.append(
            Microservice(
                id=i,
                name=f"{service_type}-{i}",
                cpu_request=cpu_req,
                memory_request=mem_req,
                service_type=service_type,
            )
        )

    # Tạo ServiceChain sau khi đã sinh xong danh sách services
    chain = ServiceChain(0, services)
    
    # Env v3:
    # Tạo traffic-weighted dependency graph.
    #
    # 1. Primary chain edges:
    #    service-0 -> service-1 -> ... -> service-N
    #    Đây là luồng chính, weight cao hơn.
    for i in range(num_services - 1):
        traffic_weight = round(float(rng.uniform(0.60, 1.00)), 2)

        chain.add_dependency(
            src=i,
            dst=i + 1,
            max_latency=5.0,
            traffic_weight=traffic_weight,
        )

    # 2. Optional cross edges:
    #    Mô phỏng các lời gọi phụ giữa các service không liền kề.
    #    Weight thấp hơn vì đây không phải luồng chính.
    if num_services >= 3:
        for i in range(num_services - 2):
            if rng.random() < 0.40:
                traffic_weight = round(float(rng.uniform(0.10, 0.50)), 2)

                chain.add_dependency(
                    src=i,
                    dst=i + 2,
                    max_latency=8.0,
                    traffic_weight=traffic_weight,
                )

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

    print("\n=== Service Chain mẫu ===")
    chain = create_sample_service_chain(5, seed=42)
    for svc in chain.services:
        print(
            f"{svc.name}: type={svc.service_type}, "
            f"CPU={svc.cpu_request}, RAM={svc.memory_request}"
        )

    print("\n=== Dependency Graph mẫu ===")
    for edge in chain.get_dependencies():
        src_name = chain.services[edge.src].name
        dst_name = chain.services[edge.dst].name

        print(
            f"{src_name} -> {dst_name}: "
            f"max_latency={edge.max_latency}, "
            f"traffic_weight={edge.traffic_weight}"
        )
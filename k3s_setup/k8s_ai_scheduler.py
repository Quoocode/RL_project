# k8s_ai_scheduler.py
from kubernetes import client, config
from stable_baselines3 import DQN
import numpy as np
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print("🔌 Đang kết nối tới Cụm K8s trên AWS Learner Lab...")
# Trỏ tới file config bạn vừa tải về
config.load_kube_config(config_file="./config_auto.yaml")
v1 = client.CoreV1Api()

print("🧠 Đang tải bộ não AI (DQN)...")
# TODO: Bạn hãy sửa lại đường dẫn này trỏ tới file model của bạn (ví dụ: ../models/ppo_model)
MODEL_PATH = "../models/dqn_hard_model.zip"
try:
    model = DQN.load(MODEL_PATH)
    print("✅ Đã nạp Model thành công!")
except Exception as e:
    print(f"⚠️ Lỗi tải model: {e}")
    print("Mẹo: Hãy trỏ đúng đường dẫn tới file .zip model mà bạn đã train.")
    exit()


def parse_quantity(value):
    """Chuyển tài nguyên K8s dạng chuỗi về số float chuẩn."""
    if value is None:
        return 0.0

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return 0.0

    if text.endswith("m"):
        return float(text[:-1]) / 1000.0

    units = {
        "Ki": 1024.0,
        "Mi": 1024.0 ** 2,
        "Gi": 1024.0 ** 3,
        "Ti": 1024.0 ** 4,
        "Pi": 1024.0 ** 5,
        "Ei": 1024.0 ** 6,
        "K": 1000.0,
        "M": 1000.0 ** 2,
        "G": 1000.0 ** 3,
        "T": 1000.0 ** 4,
        "P": 1000.0 ** 5,
        "E": 1000.0 ** 6,
    }

    for suffix, multiplier in units.items():
        if text.endswith(suffix):
            return float(text[:-len(suffix)]) * multiplier

    return float(text)


def get_worker_nodes():
    """Lấy danh sách worker nodes đang Ready trong cluster."""
    nodes_info = v1.list_node().items
    worker_nodes = []

    for node in nodes_info:
        labels = node.metadata.labels or {}
        conditions = node.status.conditions or []
        is_ready = any(condition.type == "Ready" and condition.status == "True" for condition in conditions)

        if "control-plane" in labels or "master" in labels:
            continue
        if not is_ready:
            continue

        worker_nodes.append(node)

    return worker_nodes


def get_node_load(node_name):
    """Ước lượng CPU/RAM đang dùng của node từ requests của các pod."""
    node = v1.read_node(node_name)
    allocatable = node.status.allocatable or {}
    cpu_capacity = parse_quantity(allocatable.get("cpu", 0.0))
    memory_capacity = parse_quantity(allocatable.get("memory", 0.0))

    cpu_used = 0.0
    memory_used = 0.0

    pods = v1.list_pod_for_all_namespaces(field_selector=f"spec.nodeName={node_name}").items
    for pod in pods:
        phase = getattr(pod.status, "phase", None)
        if phase in {"Succeeded", "Failed", "Unknown"}:
            continue

        for container in pod.spec.containers:
            requests = (container.resources.requests or {}) if container.resources else {}
            cpu_used += parse_quantity(requests.get("cpu", 0.0))
            memory_used += parse_quantity(requests.get("memory", 0.0))

    cpu_ratio = cpu_used / cpu_capacity if cpu_capacity > 0 else 1.0
    memory_ratio = memory_used / memory_capacity if memory_capacity > 0 else 1.0

    return cpu_ratio, memory_ratio


def build_observation(worker_nodes, current_service_idx, num_services=30, expected_nodes=5):
    """Tạo observation đúng shape như môi trường train của model."""
    obs = []

    active_nodes = worker_nodes[:expected_nodes]
    for node in active_nodes:
        cpu_ratio, memory_ratio = get_node_load(node.metadata.name)
        obs.extend([cpu_ratio, memory_ratio])

    while len(active_nodes) < expected_nodes:
        obs.extend([1.0, 1.0])
        active_nodes.append(None)

    service_onehot = np.zeros(num_services, dtype=np.float32)
    if 0 <= current_service_idx < num_services:
        service_onehot[current_service_idx] = 1.0
    obs.extend(service_onehot.tolist())

    return np.array(obs, dtype=np.float32)

def create_pod_on_node(pod_name, node_name):
    """Hàm tạo Pod và ép K8s chạy nó trên đích danh một Node cụ thể"""
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": pod_name},
        "spec": {
            "nodeName": node_name,  # ĐÂY LÀ DÒNG CODE QUYỀN LỰC NHẤT CỦA SCHEDULER!
            "containers": [{
                "name": "nginx-container",
                "image": "nginx:alpine",
                "resources": {
                    "requests": {"cpu": "100m", "memory": "100Mi"}
                }
            }]
        }
    }
    try:
        v1.create_namespaced_pod(body=pod_manifest, namespace="default")
        print(f"🚀 Thành công: Lệnh đã được gửi lên AWS!")
    except Exception as e:
        print(f"❌ Lỗi tạo Pod: {e}")

print("\n🔍 Đang quét hạ tầng Worker Nodes...")
worker_nodes = get_worker_nodes()

print(f"✅ Đã tìm thấy {len(worker_nodes)} Worker Nodes sẵn sàng nhận tải!")
print("-" * 60)

# Kịch bản Demo: Triển khai liên tục 10 Services
print("🎬 BẮT ĐẦU KỊCH BẢN TRIỂN KHAI MICROSERVICES")
for i in range(1, 11):
    pod_name = f"doan-service-{i}"
    print(f"\n📦 Có yêu cầu Service mới: [{pod_name}]")

    if not worker_nodes:
        print("❌ Không tìm thấy worker node nào đang Ready.")
        break

    current_service_idx = min(i - 1, 29)
    obs = build_observation(worker_nodes, current_service_idx)

    # AI Suy nghĩ và Đưa ra quyết định
    print("🤖 AI đang phân tích tài nguyên...")
    action, _ = model.predict(obs, deterministic=True)

    node_index = int(np.asarray(action).item())
    if node_index >= len(worker_nodes):
        load_scores = []
        for node in worker_nodes:
            cpu_ratio, memory_ratio = get_node_load(node.metadata.name)
            load_scores.append(cpu_ratio + memory_ratio)
        node_index = int(np.argmin(load_scores))

    selected_node = worker_nodes[node_index].metadata.name
    
    print(f"🎯 AI Quyết định: Đặt [{pod_name}] vào Node -> [{selected_node}]")
    
    # Giao việc cho K8s
    create_pod_on_node(pod_name, selected_node)
    
    # Chờ 3 giây để nhìn cho rõ hiệu ứng
    time.sleep(3)

print("\n🎉 KỊCH BẢN DEMO ĐÃ HOÀN TẤT!")
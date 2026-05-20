# NEXT STEPS: Dynamic 8 Nodes / 10 Services Policy

## 1. Development Goal

**Chuyển từ scenario-specific policy sang maximum-size dynamic policy**

Mục tiêu chính:
- **Nâng cấp từ fixed 5n5** → support flexible sizes up to **MAX_NODES = 8** và **MAX_SERVICES = 10**
- **Một model duy nhất** với observation/action space cố định, áp dụng được cho:
  - 5n5, 5n8, 6n7, 6n10, 8n10, hoặc bất kỳ scenario nào ≤ 8n10
- **Padding strategy**: các scenario nhỏ hơn được pad zero-vector để match cùng obs/action shape
- Vẫn sử dụng **Stable-Baselines3** (PPO, DQN, A2C) → không thay engine, chỉ refactor observation/action
- **Bước trung gian** trước khi phát triển GNN/attention-based policy trong tương lai
- **Không phá code cũ**: giữ kết quả Env v3 hiện tại, tạo Env v4 riêng

**Tại sao cần nâng cấp?**
- 5n5 quá nhỏ, mô phỏng không thực tế.
- 8n10 vẫn tractable với Stable-Baselines3 linear policy.
- Chuẩn bị cho expansion (GNN sau).
- Giúp benchmark model trên các kích thước khác nhau.

---

## 2. Target Design

### Constants

```python
# Trong envs/k8s_env.py hoặc config file mới
MAX_NODES = 8
MAX_SERVICES = 10

# Khi reset env, truyền actual_num_nodes và actual_num_services
actual_num_nodes: int    # 5-8
actual_num_services: int # 5-10
```

### Observation Design

**Observation shape cố định, không phụ thuộc vào actual_num_nodes/actual_num_services:**

```
Observation = Concatenate([
    node_features[MAX_NODES, node_feature_dim],          # [8, 5]
    service_features[MAX_SERVICES, service_feature_dim], # [10, 4]
    dependency_matrix[MAX_SERVICES, MAX_SERVICES],       # [10, 10]
    current_service_onehot[MAX_SERVICES],                # [10]
    valid_node_mask[MAX_NODES],                          # [8]
    valid_service_mask[MAX_SERVICES],                    # [10]
])

Total obs shape: 8*5 + 10*4 + 10*10 + 10 + 8 + 10 = 40 + 40 + 100 + 10 + 8 + 10 = 208
```

**Node features (5 dim per node):**
- `available_cpu_norm` = available_cpu / max_node_cpu_capacity ∈ [0, 1]
- `available_mem_norm` = available_mem / max_node_mem_capacity ∈ [0, 1]
- `cpu_capacity_norm` = cpu_capacity / max_node_cpu_capacity ∈ [0, 1]
- `mem_capacity_norm` = mem_capacity / max_node_mem_capacity ∈ [0, 1]
- `is_active` = 1.0 if active else 0.0

**Service features (4 dim per service):**
- `cpu_request_norm` = cpu_request / 2.0 ∈ [0, 1] (normalized by max expected CPU)
- `mem_request_norm` = mem_request / 4.0 ∈ [0, 1] (normalized by max expected memory)
- `service_type_id_norm` = service_type_id / num_types ∈ [0, 1]
  - hoặc onehot encoding nếu code hỗ trợ
- `is_placed` = 1.0 if placed_on >= 0 else 0.0

**Dependency matrix [MAX_SERVICES, MAX_SERVICES]:**
- `dependency_matrix[i][j]` = traffic_weight * latency_cost nếu edge (i, j) tồn tại, else 0.0
- Chỉ các service thật mới có dependency, padding service không tham gia
- Ví dụ: nếu service 5 là padding, toàn bộ hàng/cột 5 của matrix = 0

**Current service onehot [MAX_SERVICES]:**
- Luôn dài MAX_SERVICES (không phải actual_num_services)
- `onehot[current_service_idx] = 1.0`, rest = 0.0
- Nếu current_service_idx >= actual_num_services (hết service), all = 0.0

**Valid node mask [MAX_NODES]:**
- `valid_node_mask[i] = 1.0` nếu node i < actual_num_nodes, else 0.0
- Giúp model biết node nào là "thực", node nào là padding

**Valid service mask [MAX_SERVICES]:**
- `valid_service_mask[i] = 1.0` nếu service i < actual_num_services, else 0.0
- Giúp model biết service nào là "thực", service nào là padding

**Padding rule:**
- Zero-vector cho các node/service ngoài actual count
- Dependency matrix: hàng/cột padding = 0
- Onehot: chỉ có 1 bit sáng (service hiện tại), rest = 0
- Mask: 1 cho thực, 0 cho padding

---

## 3. Action Space Design

**Action space:**
```python
action_space = spaces.Discrete(MAX_NODES)  # [0, 1, 2, ..., 7]
```

**Xử lý invalid action:**

Giai đoạn đầu, không dùng action masking:

| Invalid Case | Handling | Penalty | Info |
|---|---|---|---|
| action ≥ actual_num_nodes | Đặt tạm vào action node (invalid) | -50.0 | `invalid_reason: "padding_node"` |
| node.is_active = False | Đặt tạm vào node (invalid) | -50.0 | `invalid_reason: "inactive_node"` |
| Insufficient CPU | Đặt tạm vào node (invalid) | -20.0 | `invalid_reason: "insufficient_resource"` |
| Insufficient Memory | Đặt tạm vào node (invalid) | -20.0 | `invalid_reason: "insufficient_resource"` |

**Trả về info chi tiết:**
```python
info = {
    "success": True/False,
    "invalid_action": True/False,
    "invalid_reason": "padding_node" | "inactive_node" | "insufficient_resource" | None,
    "placed_on_node": action if success else -1,
    "placed_total": count,
    "step_count": step,
}
```

**Future optimization (không làm ngay):**
- Action masking với MaskablePPO (sb3-contrib) để chỉ sample valid actions
- Với DQN, phạt invalid action vẫn tốt hơn masking
- Với PPO, masking có thể giúp hội tụ nhanh hơn

---

## 4. Recommended Implementation Strategy

### Safe Path ✅ RECOMMENDED

**Tạo class/env mới:**
```python
# Tuỳ chọn A:
class K8sPlacementEnvDynamic(gym.Env):
    """Dynamic padding cho 8n10."""
    def __init__(self, max_nodes=8, max_services=10, ...):
        self.max_nodes = max_nodes
        self.max_services = max_services
        # Tạo topology và service chain với max_nodes/max_services
        ...

# Tuỳ chọn B:
class K8sPlacementEnvV4(gym.Env):
    """Version 4: Dynamic padding."""
    ...
```

**Ưu điểm:**
- ✅ Giữ nguyên K8sPlacementEnv cũ → không phá kết quả v3
- ✅ Dễ rollback nếu lỗi
- ✅ Có thể so sánh v3 fixed vs v4 dynamic trên cùng scenario
- ✅ Training code: `env = K8sPlacementEnvDynamic(...)` thay vì `K8sPlacementEnv(...)`
- ✅ Backward compatible

**Rủi ro thấp**

### Risky Path ❌ NOT RECOMMENDED (unless very clean codebase)

**Refactor K8sPlacementEnv trực tiếp:**
```python
class K8sPlacementEnv(gym.Env):
    def __init__(self, num_nodes=5, num_services=5,
                 max_nodes=8, max_services=10, ...):
        # Thêm tham số max_nodes/max_services
        self.num_nodes = num_nodes
        self.num_services = num_services
        self.max_nodes = max_nodes if max_nodes >= num_nodes else num_nodes
        self.max_services = max_services if max_services >= num_services else num_services
        ...
```

**Rủi ro cao:**
- ❌ Phải refactor mọi nơi dùng `self.num_nodes` → `self.max_nodes`
- ❌ Khó tách tài nguyên cũ vs mới
- ❌ Nếu lỗi, khó rollback (phá code v3)
- ❌ Không nên dùng nếu deadline gần

**Khuyến nghị: Ưu tiên Safe Path**

---

## 5. Files To Modify

| File | Required Change | Priority | Risk | Effort |
|---|---|---|---|---|
| `envs/k8s_env.py` | Không sửa; tạo class mới trong file hoặc file riêng | LOW | LOW | 1h |
| `envs/topology.py` | Có thể update `create_sample_topology()` để support max_nodes tham số | MEDIUM | LOW | 30min |
| `agents/ppo_agent.py` | Thêm `--max-nodes` / `--max-services` argument | LOW | LOW | 30min |
| `agents/dqn_agent.py` | Thêm `--max-nodes` / `--max-services` argument | LOW | LOW | 30min |
| `agents/a2c_agent.py` | Thêm `--max-nodes` / `--max-services` argument | LOW | LOW | 30min |
| `train.py` | Truyền `max_nodes`, `max_services` đến agent | MEDIUM | LOW | 1h |
| `evaluate.py` | Truyền `max_nodes`, `max_services` khi eval; update baseline | MEDIUM | MEDIUM | 1.5h |
| `test_env.py` | Thêm test cho dynamic env (observation shape, padding) | HIGH | LOW | 2h |
| `test_setup.py` | Thêm test để kiểm tra 5n5 vs 5n8 vs 8n10 obs shape giống | HIGH | LOW | 2h |
| Config file (nếu có) | Thêm `max_nodes`, `max_services` | LOW | LOW | 30min |
| `GUIDE.md` hoặc README | Update documentation cho dynamic env | LOW | NONE | 1h |

**Tóm tắt:**
- Đầu tiên: Tạo K8sPlacementEnvDynamic (hoặc K8sPlacementEnvV4)
- Sau đó: Update topology.py để support creation với max_nodes
- Rồi: Update train.py, agents, evaluate.py
- Cuối: Test và document

**Total estimated effort: 8-10 giờ** (không train)

---

## 6. Implementation Checklist

### Phase 1: Add Dynamic Constants & Data Structures

- [ ] Định nghĩa `MAX_NODES = 8` và `MAX_SERVICES = 10` (global constant hoặc class attribute)
- [ ] Tách clear `actual_num_nodes` vs `max_nodes` trong constructor
- [ ] Tách clear `actual_num_services` vs `max_services` trong constructor
- [ ] Kiểm tra: observation_space phải base trên `max_nodes/max_services`, không phải `actual_num_nodes/actual_num_services`
- [ ] Verify: reset() phải chấp nhận `actual_num_nodes` và `actual_num_services` làm tham số
- [ ] Verify: topology.nodes phải có `max_nodes` phần tử (không phải `actual_num_nodes`)

### Phase 2: Observation Padding

- [ ] Pad `node_features` vector để luôn dài `5 * max_nodes` (5 features/node)
  - Node từ `actual_num_nodes` đến `max_nodes-1` được set = [0, 0, 0, 0, 0]
- [ ] Pad `service_features` vector để luôn dài `4 * max_services` (4 features/service)
  - Service từ `actual_num_services` đến `max_services-1` được set = [0, 0, 0, 0]
- [ ] Pad `dependency_matrix` về `[max_services, max_services]`
  - Hàng/cột padding = 0
  - `dependency_matrix[i][j] = 0` nếu i >= actual_num_services hoặc j >= actual_num_services
- [ ] `current_service_onehot` luôn dài `max_services`
  - Nếu `current_service_idx < actual_num_services`: onehot[current_service_idx] = 1, rest = 0
  - Nếu `current_service_idx >= actual_num_services` (hết): all = 0
- [ ] Thêm `valid_node_mask[max_nodes]`
  - `valid_node_mask[i] = 1.0` nếu i < actual_num_nodes, else 0.0
- [ ] Thêm `valid_service_mask[max_services]`
  - `valid_service_mask[i] = 1.0` nếu i < actual_num_services, else 0.0
- [ ] Observation shape check: không thay đổi giữa 5n5, 5n8, 6n10, 8n10
  - `test_obs_shape_5n5 == test_obs_shape_8n10`
- [ ] NaN/Inf check:
  - Không có NaN hoặc Inf trong observation
  - Tất cả giá trị ∈ [0, 1] hoặc [-1, 1]

### Phase 3: Action Handling

- [ ] Đổi `action_space` thành `spaces.Discrete(max_nodes)`
- [ ] Xử lý `action >= actual_num_nodes` → penalty -50, info `invalid_reason: "padding_node"`
- [ ] Xử lý `node.is_active = False` → penalty -50, info `invalid_reason: "inactive_node"`
- [ ] Xử lý `insufficient CPU` → penalty -20, info `invalid_reason: "insufficient_resource"`
- [ ] Xử lý `insufficient memory` → penalty -20, info `invalid_reason: "insufficient_resource"`
- [ ] Thêm `info["invalid_action"] = True/False`
- [ ] Thêm `info["invalid_reason"]` chi tiết
- [ ] **CRITICAL**: Invalid action KHÔNG được làm crash env, phải trả reward/terminated/truncated bình thường
- [ ] Step counter chỉ tăng 1, dù success hay invalid

### Phase 4: Dynamic Scenario Reset

- [ ] Constructor chấp nhận `actual_num_nodes` và `actual_num_services` làm tham số
- [ ] `reset()` support reset cùng scenario cố định:
  - 5n5, 5n8, 6n10, 8n10 (hardcode)
  - hoặc random scenario:
    - `actual_num_nodes ∈ [5, 8]`
    - `actual_num_services ∈ [5, 10]`
- [ ] Dependency graph chỉ tạo trên `actual_num_services` node
  - Không tạo edge nối tới padding service
- [ ] Padding service KHÔNG được tham gia vào placement:
  - `self.service_chain.services[actual_num_services:]` không bao giờ được `step()`
- [ ] Episode terminated khi `current_service_idx >= actual_num_services`
  - **KHÔNG PHẢI** khi `current_service_idx >= max_services`
- [ ] Verify: tất cả `max_steps` bảo hiểm vẫn còn (tránh infinite loop)

### Phase 5: Training Plan (Optional Phase, Only if time permits)

- [ ] Train thử **padded 5n5** (baseline comparison):
  - `env = K8sPlacementEnvDynamic(actual_num_nodes=5, actual_num_services=5, max_nodes=8, max_services=10)`
  - So sánh reward với v3 5n5 (phải tương tự)
- [ ] Train thử **padded 5n8**:
  - `env = K8sPlacementEnvDynamic(actual_num_nodes=5, actual_num_services=8, ...)`
- [ ] Train thử **dynamic random scenario** (nếu có thời gian):
  - Mỗi episode random:
    - `actual_num_nodes ∈ [5, 8]`
    - `actual_num_services ∈ [5, 10]`
- [ ] Lưu model vào `models_env_v4_dynamic_8n10/` (thư mục riêng):
  - `models_env_v4_dynamic_8n10/ppo_model/`
  - `models_env_v4_dynamic_8n10/dqn_model/`
  - `models_env_v4_dynamic_8n10/a2c_model/`
- [ ] **KHÔNG ghi đè** model v3 cũ
- [ ] Ghi log:
  - `train_reward.csv`: reward mỗi episode
  - `train_placed_rate.csv`: placement success rate
  - `train_invalid_action_rate.csv`: invalid action %
  - Tensorboard log bình thường

### Phase 6: Evaluation Plan (Optional Phase, Only if time permits)

- [ ] Evaluate model trên các scenario:
  - [ ] 5 nodes / 5 services (10 episodes)
  - [ ] 5 nodes / 8 services (10 episodes)
  - [ ] 6 nodes / 10 services (10 episodes, nếu hỗ trợ)
  - [ ] 8 nodes / 10 services stress test (10 episodes)
- [ ] Baseline so sánh:
  - [ ] Random
  - [ ] First-Fit
  - [ ] Least-Loaded (hoặc Best-Fit)
  - [ ] Round-Robin
- [ ] Metrics giữ nguyên:
  - `mean_reward`, `std_reward`
  - `placement_rate` (đặt được bao nhiêu % service)
  - `failed_placement` (số service không đặt được)
  - `invalid_action_rate` (% action invalid)
  - `bad_dependency_edges` (số edge vi phạm latency constraint)
  - `effective_latency` (latency thực tế vs expected)
  - `hotspot_count` (số node > 80% utilization)
  - `cpu_imbalance`, `memory_imbalance` (std deviation)
  - `max_cpu_utilization`, `max_mem_utilization`
- [ ] Output:
  - CSV: `results_env_v4_dynamic_8n10/evaluation_metrics.csv`
  - CSV: `results_env_v4_dynamic_8n10/evaluation_episode_metrics.csv`
  - PNG: Biểu đồ so sánh (reward, placement rate, etc.)

### Phase 7: Report Update

- [ ] Thêm section "Dynamic-size Extension" trong báo cáo hoặc GUIDE.md
- [ ] Giải thích:
  - Maximum-size observation space (8n10 với MAX_NODES=8, MAX_SERVICES=10)
  - Padding strategy (zero-vector cho thấu)
  - `valid_node_mask` và `valid_service_mask` (để model biết thực vs padding)
  - Invalid action penalty (không dùng masking)
  - Flexibility: model có thể support 5n5, 5n8, 6n10, 8n10, ...
- [ ] Ghi rõ **giới hạn**: model CHỈ hỗ trợ tối đa 8 nodes / 10 services
  - Không phải "vô hạn kích thước"
- [ ] Nêu rõ **future work**:
  - GNN/attention-based policy (không phụ thuộc vào kích thước)
  - Action masking / MaskablePPO
  - Curriculum learning (start 5n5 → 8n10)
  - Scaling study (8n10 → 16n20 vs GNN)

---

## 7. Minimal Validation Tests

**Bắt buộc chạy sau mỗi change để confirm không phá code:**

### Observation Shape Tests
- [ ] Reset env 5n5 → observation shape cố định `(208,)` hoặc bất kỳ constant nào
- [ ] Reset env 5n8 → observation shape giống 5n5
- [ ] Reset env 8n10 → observation shape giống 5n5 và 5n8
  - Kiểm tra: `obs1.shape == obs2.shape == obs3.shape`

### Invalid Action Tests
- [ ] Action = node padding (e.g., action=7 khi actual_num_nodes=5)
  - Không crash environment
  - `info["invalid_action"] = True`
  - `info["invalid_reason"] = "padding_node"`
  - Penalty được áp dụng
- [ ] Action = inactive node
  - Không crash environment
  - `info["invalid_action"] = True` (hoặc tự quyết)
- [ ] Action = insufficient resource (node đầy)
  - Không crash environment
  - Penalty được áp dụng

### Padding Service Tests
- [ ] Reset 5n5 env
- [ ] Gọi 5 bước (placement 5 service thực)
- [ ] Check `terminated = True` sau bước 5
- [ ] **KHÔNG gọi step 6 nữa** (padding service)

### Dependency Matrix Tests
- [ ] Reset 5n8 env (5 nodes, 8 services)
- [ ] Check dependency_matrix shape = `[10, 10]` (max_services)
- [ ] Check hàng/cột 8-9 (padding service) = 0
- [ ] Check hàng/cột 0-7 (service thực) có edge hoặc không tùy setup

### Mask Tests
- [ ] Reset 5n8 env
- [ ] `valid_node_mask` = `[1, 1, 1, 1, 1, 0, 0, 0]`
- [ ] `valid_service_mask` = `[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]`

### Baseline Tests (KHÔNG phải validate code, mà validate ứng dụng)
- [ ] Run Random baseline trên 5n5:
  - Không crash
  - Metrics xuất ra bình thường
- [ ] Run First-Fit baseline trên 5n8:
  - Không crash
  - **KHÔNG chọn node padding**
  - Metrics xuất ra bình thường
- [ ] Run Round-Robin baseline trên 8n10:
  - Không crash
  - Round-robin chỉ trong actual_num_nodes

### Evaluation Tests
- [ ] `evaluate.py` chạy trên 5n5 (10 episodes):
  - Xuất CSV bình thường
  - Xuất PNG biểu đồ bình thường
- [ ] `evaluate.py` chạy trên 8n10 (10 episodes):
  - Xuất CSV bình thường
  - Không crash

### Episode Length Tests
- [ ] Episode 5n5 có độ dài ≈ 5 (number of actual services)
- [ ] Episode 5n8 có độ dài ≈ 8 (không phải 10, không phải 5)
- [ ] Episode 8n10 có độ dài ≈ 10

---

## 8. Suggested Minimal Code Path

**Đường đi ít rủi ro nhất (do safe path):**

### Step 1: Create Dynamic Env Class (2 giờ)
```
→ File: envs/k8s_env.py (hoặc envs/k8s_env_v4.py)
→ Class: K8sPlacementEnvDynamic (hoặc K8sPlacementEnvV4)
→ Copy logic từ K8sPlacementEnv:
  - Constructor: thêm max_nodes, max_services
  - Topology: tạo max_nodes node
  - Service chain: tạo max_services service
  - Reward function: copy từ cũ (không thay đổi)
→ Focus vào observation builder trước
→ Test: obs shape cố định trên 5n5 và 5n8
```

### Step 2: Update Observation Builder (2 giờ)
```
→ Trong K8sPlacementEnvDynamic._get_observation():
  - Pad node_features [8 * 5] (5 feature/node)
  - Pad service_features [10 * 4] (4 feature/service)
  - Build dependency_matrix [10 * 10]
  - Onehot [10]
  - valid_node_mask [8]
  - valid_service_mask [10]
  - Concatenate all → obs [208]
→ Test: obs shape consistency
→ Test: NaN/Inf check
```

### Step 3: Update Action Space & Reward (2 giờ)
```
→ action_space = Discrete(max_nodes)
→ step() function:
  - Handle invalid actions (padding node, inactive, insufficient resource)
  - Add penalty
  - Add info dict
  - KHÔNG crash env
→ Test: invalid actions
→ Test: episode terminates at actual_num_services, not max_services
```

### Step 4: Update Reset & Topology (1.5 giờ)
```
→ Modify create_sample_topology() để accept max_nodes
→ K8sPlacementEnvDynamic.reset():
  - Support fixed scenario: reset(actual_num_nodes=5, actual_num_services=5)
  - Create dependency graph trên actual_num_services
  - Dependency matrix pad để [10, 10]
→ Test: 5n5 vs 5n8 reset bình thường
```

### Step 5: Update Training & Evaluation (2 giờ)
```
→ train.py: thêm --max-nodes, --max-services argument
→ evaluate.py: pass max_nodes, max_services vào env
→ Baseline: First-Fit, Random update để support max_nodes/actual_num_nodes
→ Test: train.py chạy được (không train lâu, chỉ 10 episodes test)
→ Test: evaluate.py chạy được
```

### Step 6: Comprehensive Tests (2 giờ)
```
→ test_env.py: thêm test cho K8sPlacementEnvDynamic
  - obs shape consistency
  - invalid actions
  - padding handling
  - mask values
→ test_setup.py: thêm test cross-scenario:
  - 5n5 vs 5n8 vs 8n10 obs shape
  - episode length = actual_num_services
→ Run all tests: commit nếu pass
```

### Step 7: Document (1 giờ)
```
→ Update GUIDE.md:
  - Explain K8sPlacementEnvDynamic
  - Explain observation padding
  - Explain how to use: env = K8sPlacementEnvDynamic(actual_num_nodes=6, actual_num_services=10, ...)
  - Explain max_nodes/max_services limit
→ Add this NEXT_STEPS_DYNAMIC_8N10.md to repo
```

**Total: ~12-14 giờ (không train)**

**Thứ tự implement:**
1. Create env class (copy cũ)
2. Update observation
3. Test observation
4. Update action space
5. Test action space
6. Update reset
7. Test reset
8. Update train/evaluate
9. Test train/evaluate
10. Comprehensive tests
11. Document
12. (Optional) Train thử

**Cách rollback nếu lỗi:**
- K8sPlacementEnv (v3 cũ) vẫn nguyên → có thể rollback bất kỳ lúc nào
- Git commit trước mỗi step
- Nếu step N fail → git reset --hard to step N-1

---

## 9. Priority Plan

### If only 1 day left (8 hours)

**Implement minimal viable dynamic env (không train):**

- [ ] Create K8sPlacementEnvDynamic class (copy từ v3)
- [ ] Update observation builder để pad đến [8, 10]
- [ ] Update action_space và step() để handle invalid actions
- [ ] Test obs shape on 5n5, 5n8, 8n10
- [ ] Update train.py để dùng env mới (phải pass max_nodes, max_services)
- [ ] Test: `python train.py --timesteps 1000 --agents ppo` (1 episode)
  - Không crash = success
- [ ] Write NEXT_STEPS_DYNAMIC_8N10.md (this file)
- [ ] Commit: "WIP: Dynamic env v4 - observation/action space only"

**Output:**
- K8sPlacementEnvDynamic dùng được (observation + action)
- Có thể train nhưng chưa đủ data point
- Document: future work cần training and evaluation

**Kỳ vọng báo cáo:**
- Giải thích được dynamic observation/action design
- Nêu kế hoạch train/eval tiếp theo
- Không cần kết quả training

### If 3 days left (24 hours)

**Implement + Basic evaluation (không train full):**

- [ ] Làm hết "1 day" checklist
- [ ] Comprehensive tests (obs, action, reset)
- [ ] Update evaluate.py baseline
- [ ] Train thử 10k timesteps trên 5n5 (test baseline)
  - Compare: v3 5n5 vs v4 5n5 (phải tương tự)
- [ ] Eval thử 5n5, 5n8 (5 episodes each)
  - Xuất metrics CSV/PNG
- [ ] Document completed phases

**Output:**
- K8sPlacementEnvDynamic ready for training
- Baseline training result on 5n5 (compare v3 vs v4)
- Eval result on 5n5 + 5n8 (verify correctness)
- CSV/PNG output files

**Kỳ vọng báo cáo:**
- V4 5n5 ~ V3 5n5 (validation that we didn't break existing)
- V4 5n8 works with dynamic observation
- Giải thích invalid action handling
- Plan for full training (8n10, random scenario)

### If 1 week left (40+ hours)

**Implement + Full training + Evaluation:**

- [ ] Làm hết "3 days" checklist
- [ ] Train PPO on:
  - [ ] Fixed 5n5 (100k steps) → compare v3
  - [ ] Fixed 5n8 (100k steps)
  - [ ] Dynamic random [5-8]n[5-10] (200k steps) → main result
- [ ] Train DQN similarly
- [ ] Train A2C similarly
- [ ] Evaluate all models on:
  - [ ] 5n5, 5n8, 6n10, 8n10 scenarios
  - [ ] Baseline: Random, First-Fit, Round-Robin
- [ ] Generate comparison plots:
  - [ ] Reward trajectory
  - [ ] Placement rate vs scenario size
  - [ ] Invalid action rate
  - [ ] CPU/Memory imbalance
- [ ] Write detailed analysis in report

**Output:**
- 3 trained models (PPO, DQN, A2C) for v4 dynamic
- Eval metrics on 4 scenarios
- Comparison plots: v3 (fixed) vs v4 (dynamic)
- Full report with analysis

**Kỳ vọng báo cáo:**
- V4 dynamic model outperform baseline
- Scaling analysis: does model work well on 8n10?
- Invalid action rate trend
- Recommendation: GNN vs padding scaling

---

## 10. Final Recommendation

### Should we develop the 8n10 dynamic direction?

**YES, with the following caveats:**

**Pro:**
1. ✅ 5n5 is too small; 8n10 is more realistic
2. ✅ Single model for [5-8]n[5-10] is useful for benchmarking
3. ✅ Safe to implement (new env class, no breaking changes)
4. ✅ Stepping stone to GNN-based policies
5. ✅ Demonstrates scalability of Stable-Baselines3 approach

**Con:**
1. ❌ Still limited to fixed max size (8n10), not truly scalable
2. ❌ Training cost increases: obs dim grows from 52 → 208 (4x)
3. ❌ Will plateau at 8n10; GNN eventually needed
4. ❌ Padding "wastes" model capacity for small scenarios

**Verdict:**
- Worth 1-2 weeks of effort
- Do NOT overcommit (avoid 1-month of training)
- Treat as stepping stone, not end goal
- If training shows ≤10% improvement over baseline → pivot to GNN
- If training shows >20% improvement → worth continuing

---

### Should we create new env class or refactor existing?

**NEW CLASS (K8sPlacementEnvDynamic or K8sPlacementEnvV4)**

✅ Reasons:
- Keeps v3 results intact
- Easy rollback
- Clear separation of concerns
- Can compare v3 vs v4 on same scenario

❌ Maintenance:
- Duplicate code (reward function, etc.)
- Need to keep both classes in sync if reward changes

**Recommendation:**
- Create new class immediately
- Plan refactor later if stable

---

### Should we keep DQN/PPO/A2C?

**YES**

✅ Reasons:
- All three agents work fine with fixed obs/action space
- Can compare agents on same task
- sb3 is mature and reliable

❌ Concerns:
- Linear policy may saturate on 8n10
- Will need GNN eventually

**Recommendation:**
- Train all 3 for comparison
- If one agent clearly wins, focus on that
- GNN can coexist with sb3 agents (compare both)

---

### Should we use action masking immediately?

**NO (not in Phase 1)**

❌ Why not:
- sb3 does NOT support action masking natively
- Would need sb3-contrib MaskablePPO (extra dependency)
- DQN + masking is tricky
- Penalty approach (current plan) is simpler and works

✅ When to add:
- After Phase 5 (after training on fixed 5n5)
- If model shows high invalid action rate (>20%)
- Can compare: penalty vs masking in separate branch

**Recommendation:**
- Start with penalty approach (Phase 3)
- Measure invalid_action_rate in logs
- Add masking in Phase 6+ if needed

---

### What to do immediately vs future work?

**IMMEDIATE (Phase 1-4, ~8 hours):**
- Create K8sPlacementEnvDynamic
- Observation + action padding
- Handle invalid actions
- Basic tests

**NEAR-TERM (Phase 5-6, ~8 hours, if time permits):**
- Training on 5n5, 5n8
- Evaluation on 4 scenarios
- CSV/PNG output

**FUTURE (not this month):**
- Full training on 8n10 random scenario
- Action masking optimization
- GNN-based policy
- Curriculum learning (5n5 → 8n10)
- Real cluster integration (k8s_real_env.py)

**Recommendation:**
- Lock in Phase 1-4 as minimum viable product
- Phase 5-6 as "nice to have"
- Phase 7+ as future research

---

### Critical constraints to preserve

**MUST DO:**
1. ✅ Do NOT delete K8sPlacementEnv (v3) code
   - Keep existing models and results
   - Can always revert if v4 fails

2. ✅ Do NOT modify reward function (Phase 3)
   - Should carry over from v3 unchanged
   - Only handle invalid action penalty addition

3. ✅ Do NOT break backward compatibility
   - `K8sPlacementEnv(num_nodes=5, num_services=5)` still works
   - New env: `K8sPlacementEnvDynamic(actual_num_nodes=5, actual_num_services=5, max_nodes=8, max_services=10)`

4. ✅ Do NOT modify evaluate.py baseline logic significantly
   - First-Fit, Random, etc. should work with Discrete(max_nodes)
   - Just need to skip padding_nodes (action >= actual_num_nodes)

5. ✅ Do NOT train until Phase 1-4 are tested and locked
   - Test first, train second
   - Saves time if we need to refactor

---

## Summary: Next 2 Weeks Roadmap

| Week | Phase | Hours | Output | Go/No-Go |
|---|---|---|---|---|
| Week 1 Day 1 | 1-3: Env class + Obs/Action | 8h | K8sPlacementEnvDynamic basic | GO if tests pass |
| Week 1 Day 2 | 4-6: Reset + Tests | 8h | Comprehensive test suite | GO if all tests pass |
| Week 1 Day 3-4 | 5-7: Training prep (Optional) | 8h | 5n5/5n8 trained, eval ready | GO if reward ≈ v3 |
| Week 2 Day 1-2 | 5-7: Full eval (Optional) | 8h | 5n5/5n8/8n10 eval, plots | GO if placement rate ↑ |
| Week 2 Day 3+ | 10: GNN research (Future) | TBD | Design notes | Defer to next sprint |

---

## Files to Create/Modify Summary

**To Create:**
- `NEXT_STEPS_DYNAMIC_8N10.md` (this file) ✅
- `envs/k8s_env_v4.py` OR extend `envs/k8s_env.py` with new class (CHOICE)
- `tests/test_env_v4.py` (comprehensive tests)

**To Modify:**
- `envs/topology.py` (add max_nodes parameter)
- `train.py` (add --max-nodes, --max-services)
- `agents/ppo_agent.py`, `agents/dqn_agent.py`, `agents/a2c_agent.py` (add arguments)
- `evaluate.py` (support max_nodes, update baseline)
- `GUIDE.md` (document dynamic env)

**Do NOT Modify:**
- `envs/k8s_env.py` (keep v3 as is, or extend without breaking)
- Existing model files or results
- Reward function logic (only add invalid action penalty)

---

**Document approved for development.**
**Status: READY TO IMPLEMENT**
**Last updated: May 18, 2026**


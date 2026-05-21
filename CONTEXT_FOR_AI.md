CONTEXT FOR EXTERNAL AI

Mục tiêu: cung cấp một bản tóm tắt ngắn gọn và có cấu trúc về toàn bộ đồ án trong workspace này để đưa làm context cho một AI khác (phân tích, lập kế hoạch, hoặc tiếp tục phát triển).

1) Tổng quan chung
- Tên dự án: Microservice Chain Placement (K8s + DRL)
- Mô tả: Sử dụng Deep Reinforcement Learning (PPO / DQN / A2C) để quyết định placement của chuỗi microservice trên cluster Kubernetes, tối ưu latency và resource utilization.
- Hướng dẫn & kiến trúc: GUIDE.md
- Thiết kế dynamic env (padding lên MAX_NODES=8, MAX_SERVICES=10): NEXT_STEPS_DYNAMIC_8N10.md

2) Entry points & scripts
- `train.py`: pipeline chính để huấn luyện và so sánh agent (supports multi-seed, export training plots).
- `evaluate.py`: script đánh giá agents và baseline (Random, First-Fit, Least-Loaded, Round-Robin). Có cơ chế fallback khi model không load được với env do obs-shape mismatch.
- `k8s_real_env.py`: bridge để chạy model lên cluster thật (sử dụng `kubectl top nodes`, deploy pod qua `kubectl apply`). Hỗ trợ dry-run.
- `run_all.py`, `create_dqn_report.py`, `plot_results.py`: các script hỗ trợ (xem trong repo).

3) Agents
- `agents/dqn_agent.py`: DQN agent; CLI; params default (learning_rate=1e-3, buffer_size=100_000, ...); save/load support; tensorboard logs to `./logs/dqn`.
- `agents/ppo_agent.py`: PPO implementation + PlacementCallback, seed helper.
- `agents/a2c_agent.py`: A2C agent (defaults tuned per results notes).

4) Environment & Topology
- `envs/k8s_env.py`: K8sPlacementEnv and dynamic variant (K8sPlacementEnvDynamic) with fixed-size observation/action using padding. Default MAX_NODES=8, MAX_SERVICES=10, obs_dim = 208.
- `envs/topology.py`: network topology, sample service chain generator, service profiles.
- Observation format: concatenation of node features, service features, dependency matrix, current service onehot, valid_node_mask, valid_service_mask (detailed in NEXT_STEPS_DYNAMIC_8N10.md).

5) Real cluster integration
- `k8s_real_env.py` reads kubectl output, builds 208-dim obs by padding (5 actual worker nodes commonly), loads DQN model and can deploy pods via `kubectl` (or dry-run). Has utilities to map node_id→node_name and to check feasibility.

6) Results & Artifacts
- Logs: `logs/ppo/`, `logs/dqn/`, `logs/a2c/` (contains per-seed subfolders in many runs).
- Models: `models/` plus variant folders (`models_env_v3_5n8s/`, `models_env_v4/`, etc).
- Results CSVs & reports: `results/` and `results_env_*` contain `evaluation_metrics.csv`, `evaluation_episode_metrics.csv`, `DQN_Results_Report.html`, and `improvements.md` with human summary.
- Key summaries: see `results/improvements.md` and `results_env_dynamic_8n10s/improvements.md` (both include current evaluation numbers and recommended next steps).

7) Current progress summary (từ `improvements.md`):
- Baseline First-Fit performs best on 5x5 scenario in reported runs.
- Example 5 nodes / 5 services (20 episodes): First-Fit mean ~19.89 ±2.91, PPO ~17.73 ±4.21, DQN ~15.71 ±9.47, A2C ~13.74 ±6.91.
- Scalability: baselines can run on 10x10; models trained on 5x5 cannot be evaluated on 10x10 due to obs-shape mismatch.
- Some retraining done (DQN/A2C 50k timesteps) and short evaluations exist.

8) Known issues & recommendations
- Observation-space mismatch across sizes: must train per-size or implement size-agnostic encoding (or use padded dynamic env consistently).
- Recommended path in repo: implement `K8sPlacementEnvDynamic` (safe path), keep old env for comparison, then train dynamic models (8n10) or curriculum train.
- Add unit tests for invalid actions, padding, and obs-shape checks; ensure evaluate fallback behavior is robust.

9) Where to look in code (important files)
- GUIDE.md
- NEXT_STEPS_DYNAMIC_8N10.md
- train.py
- evaluate.py
- k8s_real_env.py
- agents/dqn_agent.py
- agents/ppo_agent.py
- agents/a2c_agent.py
- envs/k8s_env.py
- envs/topology.py
- results/improvements.md
- results_env_dynamic_8n10s/improvements.md

10) Suggested next actions for external AI
- Option A (fast): Use this context + existing models to run `evaluate.py` on target sizes; collect CSV metrics and plots.
- Option B (medium): Train new models for 8n10 (or 10x10) following commands in `GUIDE.md`/`train.py` and save artifacts under a new models folder.
- Option C (long): Implement size-agnostic observation encoder (GNN/attention) and re-train; add Maskable action handling.

11) Quick commands (copy-paste)
```
# Train DQN (example)
python agents/dqn_agent.py --timesteps 200000 --nodes 5 --services 5 --save ./models/dqn_model

# Train full pipeline
python train.py --timesteps 100000 --agents ppo dqn a2c

# Evaluate saved model (deterministic)
python evaluate.py --episodes 20 --nodes 5 --services 5 --models-dir ./models

# Run real cluster dry-run (uses kubectl)
python k8s_real_env.py --dry-run --services 5 --model ./models/dqn_model
```

---
File 생성: `CONTEXT_FOR_AI.md` (đã tạo tại repository gốc).
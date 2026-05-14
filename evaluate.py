# evaluate.py
"""
So sánh hiệu suất các agent DRL với baseline Random và First-Fit.

Chạy mặc định (dùng model từ train.py):
    python evaluate.py

Tuỳ chỉnh:
    python evaluate.py --episodes 20 --seed 0
    python evaluate.py --nodes 5 --services 10
    python evaluate.py --models-dir ./models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")           # Không cần GUI, vẫn lưu được file
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor

from envs.k8s_env import K8sPlacementEnv
from agents.ppo_agent import set_global_seed


# ═════════════════════════════════════════════════════════════════════════════
# TẠO ENV CHUẨN
# ═════════════════════════════════════════════════════════════════════════════
def make_eval_env(num_nodes: int, num_services: int) -> K8sPlacementEnv:
    env = K8sPlacementEnv(num_nodes=num_nodes, num_services=num_services)
    return Monitor(env, filename=None)


# ═════════════════════════════════════════════════════════════════════════════
# BASELINE: RANDOM
# ═════════════════════════════════════════════════════════════════════════════
def run_random(num_nodes, num_services, n_episodes, seed) -> dict:
    env = make_eval_env(num_nodes, num_services)
    episode_metrics = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        env.action_space.seed(seed + ep)

        done = False
        ep_reward = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            done = terminated or truncated

        episode_metrics.append(
            collect_episode_metrics(
                env=env,
                ep_reward=ep_reward,
                num_services=num_services
            )
        )

    return _stats(episode_metrics=episode_metrics)


# ═════════════════════════════════════════════════════════════════════════════
# BASELINE: FIRST-FIT
# ═════════════════════════════════════════════════════════════════════════════
def run_first_fit(num_nodes, num_services, n_episodes, seed) -> dict:
    """
    First-Fit: với mỗi service, chọn node đầu tiên có đủ tài nguyên.
    Nếu không node nào fit → chọn node có utilization thấp nhất (best-effort).
    """
    env = make_eval_env(num_nodes, num_services)
    episode_metrics = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0

        while not done:
            service = env.unwrapped.service_chain.services[
                env.unwrapped.current_service_idx
            ]
            nodes = env.unwrapped.topology.nodes

            # Tìm node đầu tiên có đủ tài nguyên
            action = None
            for i, node in enumerate(nodes):
                if node.is_active and node.can_allocate(
                    service.cpu_request,
                    service.memory_request
                ):
                    action = i
                    break

            # Không node nào fit → chọn node active có utilization thấp nhất
            if action is None:
                best_util = float("inf")

                for i, node in enumerate(nodes):
                    if node.is_active:
                        util = (
                            node.cpu_used / node.cpu_capacity +
                            node.memory_used / node.memory_capacity
                        ) / 2.0

                        if util < best_util:
                            best_util = util
                            action = i

            # Tất cả node chết → chọn 0 để env xử lý phạt
            if action is None:
                action = 0

            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            done = terminated or truncated

        episode_metrics.append(
            collect_episode_metrics(
                env=env,
                ep_reward=ep_reward,
                num_services=num_services
            )
        )

    return _stats(episode_metrics=episode_metrics)

# ═════════════════════════════════════════════════════════════════════════════
# BASELINE: LEAST-LOADED
# ═════════════════════════════════════════════════════════════════════════════
def run_least_loaded(num_nodes, num_services, n_episodes, seed) -> dict:
    """
    Least-Loaded: với mỗi service, chọn node active có tải trung bình
    CPU/RAM thấp nhất mà vẫn đủ tài nguyên.

    Nếu không node nào đủ tài nguyên → chọn node active có tải thấp nhất
    để env xử lý thất bại/phạt.
    """
    env = make_eval_env(num_nodes, num_services)
    episode_metrics = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0

        while not done:
            service = env.unwrapped.service_chain.services[
                env.unwrapped.current_service_idx
            ]
            nodes = env.unwrapped.topology.nodes

            action = None
            best_score = float("inf")

            # Ưu tiên node đủ tài nguyên và có tải thấp nhất
            for i, node in enumerate(nodes):
                if not node.is_active:
                    continue

                if node.can_allocate(service.cpu_request, service.memory_request):
                    cpu_util = node.cpu_used / node.cpu_capacity
                    mem_util = node.memory_used / node.memory_capacity

                    score = (cpu_util + mem_util) / 2.0

                    if score < best_score:
                        best_score = score
                        action = i

            # Nếu không node nào đủ tài nguyên, chọn node active tải thấp nhất
            if action is None:
                best_score = float("inf")

                for i, node in enumerate(nodes):
                    if not node.is_active:
                        continue

                    cpu_util = node.cpu_used / node.cpu_capacity
                    mem_util = node.memory_used / node.memory_capacity

                    score = (cpu_util + mem_util) / 2.0

                    if score < best_score:
                        best_score = score
                        action = i

            # Nếu tất cả node chết
            if action is None:
                action = 0

            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            done = terminated or truncated

        episode_metrics.append(
            collect_episode_metrics(
                env=env,
                ep_reward=ep_reward,
                num_services=num_services
            )
        )

    return _stats(episode_metrics=episode_metrics)

# ═════════════════════════════════════════════════════════════════════════════
# BASELINE: ROUND-ROBIN
# ═════════════════════════════════════════════════════════════════════════════
def run_round_robin(num_nodes, num_services, n_episodes, seed) -> dict:
    """
    Round-Robin: lần lượt chọn node theo vòng tròn 0 → 1 → 2 → ...
    Nếu node hiện tại không đủ tài nguyên hoặc inactive, thử node tiếp theo.
    Nếu không node nào fit → vẫn chọn node theo vòng để env xử lý fail/phạt.
    """
    env = make_eval_env(num_nodes, num_services)
    episode_metrics = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0

        rr_pointer = 0

        while not done:
            service = env.unwrapped.service_chain.services[
                env.unwrapped.current_service_idx
            ]
            nodes = env.unwrapped.topology.nodes

            action = None

            # Thử lần lượt các node theo vòng tròn
            for offset in range(num_nodes):
                candidate = (rr_pointer + offset) % num_nodes
                node = nodes[candidate]

                if node.is_active and node.can_allocate(
                    service.cpu_request,
                    service.memory_request
                ):
                    action = candidate
                    break

            # Nếu không node nào fit, chọn node theo pointer hiện tại
            if action is None:
                action = rr_pointer % num_nodes

            # Cập nhật pointer cho service kế tiếp
            rr_pointer = (action + 1) % num_nodes

            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            done = terminated or truncated

        episode_metrics.append(
            collect_episode_metrics(
                env=env,
                ep_reward=ep_reward,
                num_services=num_services
            )
        )

    return _stats(episode_metrics=episode_metrics)


# ═════════════════════════════════════════════════════════════════════════════
# AI AGENTS
# ═════════════════════════════════════════════════════════════════════════════
# Map tên agent → class SB3 tương ứng
SB3_CLASS = {"ppo": PPO, "dqn": DQN, "a2c": A2C}

def run_ai_agent(name: str, model_path: str,
                 num_nodes, num_services, n_episodes, seed) -> dict:
    """
    Load model đã train và chạy evaluate.
    deterministic=True: agent luôn chọn action tốt nhất theo policy đã học.
    """
    env = make_eval_env(num_nodes, num_services)
    SB3Class = SB3_CLASS[name]

    model = SB3Class.load(model_path, env=env)

    episode_metrics = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            # SB3 có thể trả action dạng numpy.ndarray
            # Env cần int để placed_on không bị lưu thành ndarray
            action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            done = terminated or truncated

        episode_metrics.append(
            collect_episode_metrics(
                env=env,
                ep_reward=ep_reward,
                num_services=num_services
            )
        )

    return _stats(episode_metrics=episode_metrics)

# ═════════════════════════════════════════════════════════════════════════════
# EPISODE METRICS
# ═════════════════════════════════════════════════════════════════════════════
def collect_episode_metrics(env, ep_reward: float, num_services: int) -> dict:
    """
    Thu thập các metrics sau khi một episode kết thúc.

    Mục tiêu:
    - Không chỉ đo reward và placement rate
    - Mà còn đo load balance, hotspot, latency cost, số node được dùng
    """
    raw_env = env.unwrapped
    nodes = raw_env.topology.nodes
    services = raw_env.service_chain.services

    cpu_utils = np.array([
        node.cpu_used / node.cpu_capacity if node.is_active else 1.0
        for node in nodes
    ])

    mem_utils = np.array([
        node.memory_used / node.memory_capacity if node.is_active else 1.0
        for node in nodes
    ])

    placed_services = [
        svc for svc in services
        if svc.placed_on >= 0
    ]

    placed_count = len(placed_services)
    failed_count = num_services - placed_count

    used_nodes = len(set(
        svc.placed_on for svc in placed_services
    ))

    # Tính latency cost giữa các service liên tiếp trong chain
    total_latency = 0.0
    latency_edges = 0

    for i in range(1, len(services)):
        prev_svc = services[i - 1]
        cur_svc = services[i]

        if prev_svc.placed_on >= 0 and cur_svc.placed_on >= 0:
            latency = raw_env.topology.get_latency(
                prev_svc.placed_on,
                cur_svc.placed_on
            )

            if np.isfinite(latency):
                total_latency += latency
                latency_edges += 1

    avg_latency = (
        total_latency / latency_edges
        if latency_edges > 0
        else 0.0
    )

    return {
        "reward": float(ep_reward),
        "placed_rate": placed_count / num_services,
        "failed_count": int(failed_count),
        "used_nodes": int(used_nodes),
        "max_cpu_util": float(np.max(cpu_utils)),
        "max_mem_util": float(np.max(mem_utils)),
        "avg_cpu_util": float(np.mean(cpu_utils)),
        "avg_mem_util": float(np.mean(mem_utils)),
        "cpu_imbalance": float(np.std(cpu_utils)),
        "mem_imbalance": float(np.std(mem_utils)),
        "hotspot_count": int(
            np.sum((cpu_utils > 0.8) | (mem_utils > 0.8))
        ),
        "total_latency_cost": float(total_latency),
        "avg_latency_cost": float(avg_latency),
    }


# ═════════════════════════════════════════════════════════════════════════════
# HELPER STATS
# ═════════════════════════════════════════════════════════════════════════════
def _stats(rewards: list = None,
           placed_rates: list = None,
           episode_metrics: list = None) -> dict:
    """
    Tổng hợp kết quả evaluate.

    Hỗ trợ 2 kiểu:
    1. Kiểu cũ: rewards + placed_rates
    2. Kiểu mới: episode_metrics chứa nhiều metric hơn
    """

    # ── Kiểu mới: dùng episode_metrics ─────────────────────────────────────
    if episode_metrics is not None:
        rewards = [m["reward"] for m in episode_metrics]

        arr = np.array(rewards)

        def mean_metric(key: str) -> float:
            return float(np.mean([m[key] for m in episode_metrics]))

        return {
            # Reward metrics
            "mean"                : float(np.mean(arr)),
            "std"                 : float(np.std(arr)),
            "max"                 : float(np.max(arr)),
            "min"                 : float(np.min(arr)),

            # Placement metrics
            "placed_mean"         : mean_metric("placed_rate"),
            "failed_mean"         : mean_metric("failed_count"),
            "used_nodes_mean"     : mean_metric("used_nodes"),

            # Resource utilization metrics
            "avg_cpu_util_mean"   : mean_metric("avg_cpu_util"),
            "avg_mem_util_mean"   : mean_metric("avg_mem_util"),
            "max_cpu_util_mean"   : mean_metric("max_cpu_util"),
            "max_mem_util_mean"   : mean_metric("max_mem_util"),
            "cpu_imbalance_mean"  : mean_metric("cpu_imbalance"),
            "mem_imbalance_mean"  : mean_metric("mem_imbalance"),
            "hotspot_mean"        : mean_metric("hotspot_count"),

            # Latency metrics
            "total_latency_mean"  : mean_metric("total_latency_cost"),
            "avg_latency_mean"    : mean_metric("avg_latency_cost"),
        }

    # ── Kiểu cũ: giữ tương thích với code hiện tại ─────────────────────────
    arr = np.array(rewards)

    return {
        "mean"        : float(np.mean(arr)),
        "std"         : float(np.std(arr)),
        "max"         : float(np.max(arr)),
        "min"         : float(np.min(arr)),
        "placed_mean" : float(np.mean(placed_rates)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# BIỂU ĐỒ
# ═════════════════════════════════════════════════════════════════════════════
def draw_chart(labels: list, results: list[dict],
               num_nodes: int, num_services: int, save_dir: str):
    means = [r["mean"]        for r in results]
    stds  = [r["std"]         for r in results]
    rates = [r["placed_mean"] for r in results]

    colors = [
    "#e74c3c",  # Random
    "#f39c12",  # First-Fit
    "#1abc9c",  # Least-Loaded
    "#95a5a6",  # Round-Robin
    "#3498db",  # PPO
    "#2ecc71",  # DQN
    "#9b59b6",  # A2C
    "#34495e",  # fallback
    "#e67e22",  # fallback
]

    active_colors = colors[:len(labels)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"So sánh hiệu suất — {num_nodes} nodes / {num_services} services",
        fontsize=15, fontweight="bold"
    )

    # ── Biểu đồ 1: Mean reward ± std ───────────────────────────────────────
    bars = ax1.bar(labels, means, color=active_colors, width=0.55,
                   yerr=stds, capsize=5, error_kw={"linewidth": 1.5})

    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(stds) * 0.1 + 1,
                 f"{mean:.1f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    ax1.set_title("Mean Reward (± Std)", fontsize=12)
    ax1.set_ylabel("Reward")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    lo = min(m - s for m, s in zip(means, stds))
    hi = max(m + s for m, s in zip(means, stds))
    margin = (hi - lo) * 0.2 or 5
    ax1.set_ylim(min(lo - margin, -5), hi + margin * 2)

    # ── Biểu đồ 2: Tỉ lệ service đặt thành công ────────────────────────────
    bars2 = ax2.bar(labels, [r * 100 for r in rates],
                    color=active_colors, width=0.55)

    for bar, rate in zip(bars2, rates):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f"{rate*100:.1f}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    ax2.set_title("Service Placement Rate", fontsize=12)
    ax2.set_ylabel("Tỉ lệ đặt thành công (%)")
    ax2.set_ylim(0, 115)
    ax2.axhline(100, color="green", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "comparison_chart.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\n  ✅ Đã lưu biểu đồ: {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# IN BẢNG KẾT QUẢ
# ═════════════════════════════════════════════════════════════════════════════
def print_table(labels: list, results: list[dict]):
    print(f"\n{'═'*90}")
    print(f"  KẾT QUẢ EVALUATE — OVERALL")
    print(f"{'═'*90}")
    print(
        f"  {'Agent':<14} "
        f"{'Mean':>8} {'Std':>7} {'Max':>8} {'Min':>8} "
        f"{'Placed%':>8} {'Fail':>6} {'UsedN':>6}"
    )
    print(f"  {'-'*86}")

    best_idx = int(np.argmax([r["mean"] for r in results]))

    for i, (label, r) in enumerate(zip(labels, results)):
        marker = " 🏆" if i == best_idx else ""

        print(
            f"  {label:<14} "
            f"{r['mean']:>8.2f} "
            f"{r['std']:>7.2f} "
            f"{r['max']:>8.2f} "
            f"{r['min']:>8.2f} "
            f"{r['placed_mean']*100:>7.1f}% "
            f"{r.get('failed_mean', 0.0):>6.2f} "
            f"{r.get('used_nodes_mean', 0.0):>6.2f}"
            f"{marker}"
        )

    print(f"{'═'*90}")

    print(f"\n{'═'*90}")
    print(f"  KẾT QUẢ EVALUATE — RESOURCE / LATENCY DETAILS")
    print(f"{'═'*90}")
    print(
        f"  {'Agent':<14} "
        f"{'CPUstd':>7} {'MEMstd':>7} "
        f"{'MaxCPU':>7} {'MaxMEM':>7} "
        f"{'Hotspot':>8} {'TotLat':>8} {'AvgLat':>8}"
    )
    print(f"  {'-'*86}")

    for label, r in zip(labels, results):
        print(
            f"  {label:<14} "
            f"{r.get('cpu_imbalance_mean', 0.0):>7.3f} "
            f"{r.get('mem_imbalance_mean', 0.0):>7.3f} "
            f"{r.get('max_cpu_util_mean', 0.0):>7.3f} "
            f"{r.get('max_mem_util_mean', 0.0):>7.3f} "
            f"{r.get('hotspot_mean', 0.0):>8.2f} "
            f"{r.get('total_latency_mean', 0.0):>8.2f} "
            f"{r.get('avg_latency_mean', 0.0):>8.2f}"
        )

    print(f"{'═'*90}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate và so sánh các agent"
    )
    parser.add_argument("--episodes",    type=int, default=10)
    parser.add_argument("--seed",        type=int, default=999)
    parser.add_argument("--nodes",       type=int, default=5)
    parser.add_argument("--services",    type=int, default=5)
    parser.add_argument("--models-dir",  type=str, default="./models")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument(
        "--agents", nargs="+", choices=["ppo", "dqn", "a2c"],
        default=["ppo", "dqn", "a2c"],
        help="AI agent cần evaluate (default: tất cả)"
    )
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()
    set_global_seed(args.seed)

    N  = args.nodes
    S  = args.services
    EP = args.episodes

    print(f"\n{'═'*55}")
    print(f"  EVALUATE — {N} nodes | {S} services | {EP} episodes")
    print(f"{'═'*55}")

    labels  = []
    results = []

    # ── Baseline Random ──────────────────────────────────────────────────────
    print("\n  [1/7] Random baseline...")
    labels.append("Random")
    results.append(run_random(N, S, EP, args.seed))
    r = results[-1]
    print(f"        Mean: {r['mean']:.2f} ± {r['std']:.2f} | "
          f"Placed: {r['placed_mean']*100:.1f}%")

    # ── Baseline First-Fit ───────────────────────────────────────────────────
    print("\n  [2/7] First-Fit baseline...")
    labels.append("First-Fit")
    results.append(run_first_fit(N, S, EP, args.seed))
    r = results[-1]
    print(f"        Mean: {r['mean']:.2f} ± {r['std']:.2f} | "
          f"Placed: {r['placed_mean']*100:.1f}%")
    
    # ── Baseline Least-Loaded ────────────────────────────────────────────────
    print("\n  [3/7] Least-Loaded baseline...")
    labels.append("Least-Loaded")
    results.append(run_least_loaded(N, S, EP, args.seed))
    r = results[-1]
    print(f"        Mean: {r['mean']:.2f} ± {r['std']:.2f} | "
        f"Placed: {r['placed_mean']*100:.1f}%")
    
    # ── Baseline Round-Robin ─────────────────────────────────────────────────
    print("\n  [4/7] Round-Robin baseline...")
    labels.append("Round-Robin")
    results.append(run_round_robin(N, S, EP, args.seed))
    r = results[-1]
    print(f"        Mean: {r['mean']:.2f} ± {r['std']:.2f} | "
        f"Placed: {r['placed_mean']*100:.1f}%")

    # ── AI Agents ────────────────────────────────────────────────────────────
    agent_names = {"ppo": "PPO", "dqn": "DQN", "a2c": "A2C"}
    step = 3
    for name in args.agents:
        model_path = os.path.join(args.models_dir, f"{name}_model")
        print(f"\n  [{step}/5] {name.upper()} agent — {model_path}.zip ...")
        step += 1
        try:
            labels.append(agent_names[name])
            results.append(
                run_ai_agent(name, model_path, N, S, EP, args.seed)
            )
            r = results[-1]
            print(f"        Mean: {r['mean']:.2f} ± {r['std']:.2f} | "
                  f"Placed: {r['placed_mean']*100:.1f}%")
        except FileNotFoundError:
            print(f"        ⚠️  Không tìm thấy {model_path}.zip — bỏ qua.")
            labels.pop()  # xoá label vừa thêm
        except Exception as e:
            print(f"        ❌ Lỗi: {e}")
            labels.pop()

    # ── Kết quả ──────────────────────────────────────────────────────────────
    print_table(labels, results)
    draw_chart(labels, results, N, S, args.results_dir)

    print(f"\n  Chạy xong. Biểu đồ lưu tại: {args.results_dir}/comparison_chart.png")

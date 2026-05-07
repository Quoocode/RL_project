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
    rewards, placed_rates = [], []

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
        rewards.append(ep_reward)
        placed_rates.append(info["placed_total"] / num_services)

    return _stats(rewards, placed_rates)


# ═════════════════════════════════════════════════════════════════════════════
# BASELINE: FIRST-FIT
# ═════════════════════════════════════════════════════════════════════════════
def run_first_fit(num_nodes, num_services, n_episodes, seed) -> dict:
    """
    First-Fit: với mỗi service, chọn node đầu tiên có đủ tài nguyên.
    Nếu không node nào fit → chọn node có utilization thấp nhất (best-effort).
    """
    env = make_eval_env(num_nodes, num_services)
    rewards, placed_rates = [], []

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
                    service.cpu_request, service.memory_request
                ):
                    action = i
                    break

            # Không node nào fit → chọn node active có utilization thấp nhất
            if action is None:
                best_util = float("inf")
                for i, node in enumerate(nodes):
                    if node.is_active:
                        util = (node.cpu_used / node.cpu_capacity +
                                node.memory_used / node.memory_capacity) / 2
                        if util < best_util:
                            best_util = util
                            action = i

            # Tất cả chết → chọn 0 (sẽ bị phạt)
            if action is None:
                action = 0

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)
        placed_rates.append(info["placed_total"] / num_services)

    return _stats(rewards, placed_rates)


# ═════════════════════════════════════════════════════════════════════════════
# AI AGENTS
# ═════════════════════════════════════════════════════════════════════════════
# Map tên agent → class SB3 tương ứng
SB3_CLASS = {"ppo": PPO, "dqn": DQN, "a2c": A2C}

def run_ai_agent(name: str, model_path: str,
                 num_nodes, num_services, n_episodes, seed) -> dict:
    """
    Load model đã train và chạy evaluate.
    FIX BUG CŨ: dùng SB3Class.load(path, env=env) thay vì
    agent.model = agent.model.load(path) — cách cũ không bind env,
    predict sẽ dùng sai observation space.
    """
    env = make_eval_env(num_nodes, num_services)
    SB3Class = SB3_CLASS[name]

    # ── FIX CHÍNH ──────────────────────────────────────────────────────────
    model = SB3Class.load(model_path, env=env)
    # ───────────────────────────────────────────────────────────────────────

    rewards, placed_rates = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0

        while not done:
            # deterministic=True: luôn chọn action tốt nhất, không random
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)
        placed_rates.append(info["placed_total"] / num_services)

    return _stats(rewards, placed_rates)


# ═════════════════════════════════════════════════════════════════════════════
# HELPER STATS
# ═════════════════════════════════════════════════════════════════════════════
def _stats(rewards: list, placed_rates: list) -> dict:
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

    colors = ["#e74c3c", "#f39c12", "#3498db", "#9b59b6", "#2ecc71"]
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
    print(f"\n{'═'*65}")
    print(f"  KẾT QUẢ EVALUATE")
    print(f"{'═'*65}")
    print(f"  {'Agent':<18} {'Mean':>8} {'Std':>7} {'Max':>8} "
          f"{'Min':>8} {'Placed%':>8}")
    print(f"  {'-'*61}")
    best_idx = int(np.argmax([r["mean"] for r in results]))
    for i, (label, r) in enumerate(zip(labels, results)):
        marker = " 🏆" if i == best_idx else ""
        print(f"  {label:<18} {r['mean']:>8.2f} {r['std']:>7.2f} "
              f"{r['max']:>8.2f} {r['min']:>8.2f} "
              f"{r['placed_mean']*100:>7.1f}%{marker}")
    print(f"{'═'*65}")


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
    print("\n  [1/5] Random baseline...")
    labels.append("Random")
    results.append(run_random(N, S, EP, args.seed))
    r = results[-1]
    print(f"        Mean: {r['mean']:.2f} ± {r['std']:.2f} | "
          f"Placed: {r['placed_mean']*100:.1f}%")

    # ── Baseline First-Fit ───────────────────────────────────────────────────
    print("\n  [2/5] First-Fit baseline...")
    labels.append("First-Fit")
    results.append(run_first_fit(N, S, EP, args.seed))
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

# train.py
"""
Entry point chính để train và so sánh các DRL agent.

Chạy tất cả agent (mặc định):
    python train.py

Chọn agent cụ thể:
    python train.py --agents ppo dqn
    python train.py --agents a2c

Tuỳ chỉnh:
    python train.py --timesteps 200000 --seed 0 --nodes 5 --services 5
    python train.py --agents ppo --timesteps 500000 --eval-episodes 20
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import traceback
import numpy as np

from stable_baselines3.common.monitor import Monitor

from envs.k8s_env import K8sPlacementEnv
from agents.ppo_agent import PPOAgent, set_global_seed
from agents.dqn_agent import DQNAgent
from agents.a2c_agent import A2CAgent


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
def evaluate_agent(agent, num_nodes: int, num_services: int,
                   n_episodes: int = 10, seed: int = 42) -> dict:
    """
    Chạy agent (deterministic) qua n_episodes episode và trả về thống kê.
    Tạo env riêng để eval, không ảnh hưởng env đang train.
    """
    eval_env = K8sPlacementEnv(num_nodes=num_nodes, num_services=num_services)
    eval_env = Monitor(eval_env, filename=None)

    rewards      = []
    placed_rates = []   # Tỉ lệ service đặt thành công mỗi episode

    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)
        placed_rates.append(info["placed_total"] / num_services)

    return {
        "mean_reward"  : float(np.mean(rewards)),
        "std_reward"   : float(np.std(rewards)),
        "max_reward"   : float(np.max(rewards)),
        "min_reward"   : float(np.min(rewards)),
        "mean_placed"  : float(np.mean(placed_rates)),   # 1.0 = đặt hết services
    }


# ═════════════════════════════════════════════════════════════════════════════
# TRAIN MỘT AGENT
# ═════════════════════════════════════════════════════════════════════════════
AGENT_MAP = {
    "ppo": (PPOAgent, "./logs/ppo", "./models/ppo_model"),
    "dqn": (DQNAgent, "./logs/dqn", "./models/dqn_model"),
    "a2c": (A2CAgent, "./logs/a2c", "./models/a2c_model"),
}

def train_one(name: str, args) -> dict | None:
    """
    Train một agent, eval sau khi xong, trả về kết quả.
    Bắt exception để nếu 1 agent lỗi, các agent khác vẫn chạy tiếp.
    """
    AgentClass, log_dir, save_path = AGENT_MAP[name]

    print(f"\n{'#'*55}")
    print(f"#  [{name.upper()}] BẮT ĐẦU")
    print(f"{'#'*55}")

    try:
        # Tạo env mới cho mỗi agent — seed cố định để so sánh công bằng
        set_global_seed(args.seed)
        env = K8sPlacementEnv(num_nodes=args.nodes, num_services=args.services)
        env = Monitor(env, filename=None)
        env.reset(seed=args.seed)

        agent = AgentClass(env, seed=args.seed, tensorboard_log=log_dir)

        t0 = time.time()
        agent.train(
            total_timesteps = args.timesteps,
            log_interval    = args.log_interval,
            save_path       = save_path,
        )
        train_time = time.time() - t0

        # Eval sau khi train xong
        print(f"\n  Đang eval {name.upper()} ({args.eval_episodes} episodes)...")
        stats = evaluate_agent(
            agent,
            num_nodes    = args.nodes,
            num_services = args.services,
            n_episodes   = args.eval_episodes,
            seed         = args.seed + 1000,   # Seed khác để tránh trùng train
        )
        stats["train_time"] = train_time
        stats["status"]     = "OK"

        print(f"  Eval xong | Mean reward: {stats['mean_reward']:.2f} "
              f"| Placed: {stats['mean_placed']*100:.1f}%")
        return stats

    except Exception:
        print(f"\n  ❌ [{name.upper()}] FAILED:")
        traceback.print_exc()
        return {"status": "FAILED"}


# ═════════════════════════════════════════════════════════════════════════════
# BẢNG SO SÁNH
# ═════════════════════════════════════════════════════════════════════════════
def print_summary(results: dict, args):
    """In bảng tổng kết so sánh các agent sau khi tất cả train xong."""

    print(f"\n\n{'═'*65}")
    print(f"  KẾT QUẢ TỔNG HỢP")
    print(f"  Config: {args.nodes} nodes | {args.services} services | "
          f"seed={args.seed} | {args.timesteps:,} timesteps")
    print(f"{'═'*65}")
    print(f"  {'Agent':<8} {'Mean±Std':>14} {'Max':>8} {'Min':>8} "
          f"{'Placed%':>8} {'Time':>8}  Status")
    print(f"  {'-'*61}")

    best_name   = None
    best_reward = float("-inf")

    for name, stats in results.items():
        if stats["status"] != "OK":
            print(f"  {name.upper():<8} {'—':>14} {'—':>8} {'—':>8} "
                  f"{'—':>8} {'—':>8}  ❌ FAILED")
            continue

        mean  = stats["mean_reward"]
        std   = stats["std_reward"]
        mx    = stats["max_reward"]
        mn    = stats["min_reward"]
        place = stats["mean_placed"] * 100
        t     = stats["train_time"]

        mins, secs = divmod(int(t), 60)
        time_str = f"{mins}m{secs:02d}s"

        print(f"  {name.upper():<8} {mean:>7.2f}±{std:<6.2f} "
              f"{mx:>8.2f} {mn:>8.2f} {place:>7.1f}% {time_str:>8}")

        if mean > best_reward:
            best_reward = mean
            best_name   = name

    print(f"{'═'*65}")

    if best_name:
        print(f"  🏆 Agent tốt nhất: {best_name.upper()} "
              f"(mean reward = {best_reward:.2f})")

    print(f"\n  Xem chi tiết trên Tensorboard:")
    for name in results:
        _, log_dir, _ = AGENT_MAP[name]
        print(f"    tensorboard --logdir {log_dir}   ({name.upper()})")
    print(f"  Hoặc so sánh tất cả cùng lúc:")
    trained = list(results.keys())
    log_arg = ",".join(f"{n}:{AGENT_MAP[n][1]}" for n in trained)
    print(f"    tensorboard --logdir_spec {log_arg}")
    print(f"{'═'*65}\n")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train & so sánh DRL agents cho K8s Microservice Placement"
    )
    parser.add_argument(
        "--agents", nargs="+", choices=["ppo", "dqn", "a2c"], default=["ppo", "dqn", "a2c"],
        help="Agent cần train (default: tất cả)"
    )
    parser.add_argument("--timesteps",      type=int, default=100_000,
                        help="Timestep mỗi agent (default: 100000)")
    parser.add_argument("--seed",           type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--nodes",          type=int, default=5,
                        help="Số node (default: 5)")
    parser.add_argument("--services",       type=int, default=5,
                        help="Số services (default: 5)")
    parser.add_argument("--eval-episodes",  type=int, default=10,
                        help="Số episode eval sau train (default: 10)")
    parser.add_argument("--log-interval",   type=int, default=100,
                        help="In terminal mỗi N episode (default: 100)")
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()

    print(f"\n{'═'*55}")
    print(f"  K8s MICROSERVICE PLACEMENT — DRL TRAINING")
    print(f"{'═'*55}")
    print(f"  Agents    : {[a.upper() for a in args.agents]}")
    print(f"  Timesteps : {args.timesteps:,} mỗi agent")
    print(f"  Seed      : {args.seed}")
    print(f"  Cluster   : {args.nodes} nodes | {args.services} services")
    print(f"{'═'*55}")

    results = {}
    total_t0 = time.time()

    for name in args.agents:
        results[name] = train_one(name, args)

    total_time = time.time() - total_t0
    mins, secs = divmod(int(total_time), 60)
    print(f"\n  Tổng thời gian: {mins}m{secs:02d}s")

    print_summary(results, args)

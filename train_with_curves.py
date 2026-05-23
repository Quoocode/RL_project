# train_with_curves.py
"""
Train and compare DRL agents for K8s Microservice Placement.

Main use case for the report:
    python train_with_curves.py --agents ppo dqn a2c --nodes 5 --services 8 --timesteps 300000 --seed 12345 --eval-episodes 50 --experiment-dir experiment_5n8s_300k_learning

Outputs:
    <experiment_dir>/
    ├── models/
    ├── tensorboard/
    ├── monitor/
    ├── training_curves/
    ├── eval/
    ├── training_summary.csv
    └── experiment_config.json
"""

import sys
import os

# Avoid UnicodeEncodeError on Windows PowerShell/cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
import json
import time
import traceback
from datetime import datetime
from typing import Optional

import numpy as np

from stable_baselines3.common.monitor import Monitor

from envs.k8s_env import K8sPlacementEnv
from agents.ppo_agent import PPOAgent, set_global_seed
from agents.dqn_agent import DQNAgent
from agents.a2c_agent import A2CAgent


AGENT_CLASSES = {
    "ppo": PPOAgent,
    "dqn": DQNAgent,
    "a2c": A2CAgent,
}


# =============================================================================
# Utility functions
# =============================================================================

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def default_experiment_dir(args) -> str:
    k_steps = args.timesteps // 1000
    return f"experiment_{args.nodes}n{args.services}s_{k_steps}k_seed{args.seed}"


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_csv(path: str, row: dict, fieldnames: list[str]) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_rows_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    window = max(1, int(window))
    out = []
    cumsum = 0.0
    queue = []
    for v in values:
        queue.append(v)
        cumsum += v
        if len(queue) > window:
            cumsum -= queue.pop(0)
        out.append(cumsum / len(queue))
    return out


def read_monitor_csv(path: str) -> list[dict]:
    """
    Read SB3 Monitor CSV.

    Monitor format:
        #{"t_start": ...}
        r,l,t
        ...
    """
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        non_comment_lines = [line for line in f if not line.startswith("#")]

    if not non_comment_lines:
        return []

    reader = csv.DictReader(non_comment_lines)
    for idx, row in enumerate(reader, start=1):
        try:
            rows.append({
                "episode": idx,
                "reward": float(row["r"]),
                "length": int(float(row["l"])),
                "time": float(row["t"]),
            })
        except Exception:
            continue

    return rows


def save_processed_monitor_csv(
    monitor_path: str,
    output_csv: str,
    smooth_window: int,
) -> list[dict]:
    rows = read_monitor_csv(monitor_path)
    rewards = [r["reward"] for r in rows]
    rolling = moving_average(rewards, smooth_window)

    processed = []
    for row, roll in zip(rows, rolling):
        processed.append({
            "episode": row["episode"],
            "reward": row["reward"],
            "rolling_reward": roll,
            "length": row["length"],
            "time": row["time"],
        })

    write_rows_csv(
        output_csv,
        processed,
        ["episode", "reward", "rolling_reward", "length", "time"],
    )
    return processed


def make_learning_curve_plots(
    processed_by_agent: dict,
    figures_dir: str,
    smooth_window: int,
) -> None:
    """
    Generate:
      - combined_learning_curve.png
      - learning_curve_<agent>.png
      - combined_episode_length_curve.png
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Cannot import matplotlib, skip plots: {exc}")
        return

    ensure_dir(figures_dir)

    # Combined reward curve
    plt.figure(figsize=(10, 6))
    has_data = False
    for agent_name, rows in processed_by_agent.items():
        if not rows:
            continue
        has_data = True
        episodes = [r["episode"] for r in rows]
        rolling = [r["rolling_reward"] for r in rows]
        plt.plot(episodes, rolling, label=agent_name.upper())

    if has_data:
        plt.xlabel("Episode")
        plt.ylabel(f"Episode reward (moving average, window={smooth_window})")
        plt.title("Training learning curve - reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "combined_learning_curve.png"), dpi=200)
    plt.close()

    # Individual reward curves
    for agent_name, rows in processed_by_agent.items():
        if not rows:
            continue
        episodes = [r["episode"] for r in rows]
        rewards = [r["reward"] for r in rows]
        rolling = [r["rolling_reward"] for r in rows]

        plt.figure(figsize=(10, 6))
        plt.plot(episodes, rewards, alpha=0.25, label="raw episode reward")
        plt.plot(episodes, rolling, label=f"moving average ({smooth_window})")
        plt.xlabel("Episode")
        plt.ylabel("Episode reward")
        plt.title(f"{agent_name.upper()} training learning curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(figures_dir, f"learning_curve_{agent_name}.png"),
            dpi=200,
        )
        plt.close()

    # Combined episode length curve
    plt.figure(figsize=(10, 6))
    has_data = False
    for agent_name, rows in processed_by_agent.items():
        if not rows:
            continue
        has_data = True
        episodes = [r["episode"] for r in rows]
        lengths = [r["length"] for r in rows]
        rolling_len = moving_average(lengths, smooth_window)
        plt.plot(episodes, rolling_len, label=agent_name.upper())

    if has_data:
        plt.xlabel("Episode")
        plt.ylabel(f"Episode length (moving average, window={smooth_window})")
        plt.title("Training curve - episode length")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(figures_dir, "combined_episode_length_curve.png"),
            dpi=200,
        )
    plt.close()


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_agent(
    agent,
    num_nodes: int,
    num_services: int,
    n_episodes: int = 10,
    seed: int = 42,
    eval_csv_path: Optional[str] = None,
) -> dict:
    """
    Evaluate deterministic agent and return summary statistics.
    Also optionally saves per-episode evaluation results.
    """
    eval_env = K8sPlacementEnv(num_nodes=num_nodes, num_services=num_services)
    eval_env = Monitor(eval_env, filename=None)

    episode_rows = []
    rewards = []
    placed_rates = []

    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        final_info = {}

        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += float(reward)
            final_info = info
            done = terminated or truncated

        placed_total = final_info.get("placed_total", 0)
        placed_rate = placed_total / num_services if num_services > 0 else 0.0

        rewards.append(ep_reward)
        placed_rates.append(placed_rate)

        episode_rows.append({
            "episode": ep,
            "seed": seed + ep,
            "reward": ep_reward,
            "placed_total": placed_total,
            "num_services": num_services,
            "placed_rate": placed_rate,
        })

    if eval_csv_path:
        write_rows_csv(
            eval_csv_path,
            episode_rows,
            [
                "episode",
                "seed",
                "reward",
                "placed_total",
                "num_services",
                "placed_rate",
            ],
        )

    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "max_reward": float(np.max(rewards)) if rewards else 0.0,
        "min_reward": float(np.min(rewards)) if rewards else 0.0,
        "mean_placed": float(np.mean(placed_rates)) if placed_rates else 0.0,
        "eval_episode_csv": eval_csv_path or "",
    }


# =============================================================================
# Train one agent
# =============================================================================

def train_one(name: str, args, paths: dict) -> dict:
    """
    Train one agent, evaluate it, and save training artifacts.
    """
    AgentClass = AGENT_CLASSES[name]

    model_path = os.path.join(paths["models_dir"], f"{name}_model")
    tensorboard_log_dir = os.path.join(paths["tensorboard_dir"], name)
    monitor_path = os.path.join(paths["monitor_dir"], f"{name}.monitor.csv")
    processed_monitor_csv = os.path.join(
        paths["monitor_dir"],
        f"{name}_training_episodes.csv",
    )
    eval_csv_path = os.path.join(paths["eval_dir"], f"{name}_eval_episodes.csv")

    ensure_dir(tensorboard_log_dir)

    print("\n" + "#" * 70)
    print(f"# [{name.upper()}] START")
    print("#" * 70)
    print(f"Model path      : {model_path}")
    print(f"TensorBoard log : {tensorboard_log_dir}")
    print(f"Monitor CSV     : {monitor_path}")

    try:
        set_global_seed(args.seed)

        env = K8sPlacementEnv(
            num_nodes=args.nodes,
            num_services=args.services,
        )
        env = Monitor(env, filename=monitor_path)
        env.reset(seed=args.seed)

        agent = AgentClass(
            env,
            seed=args.seed,
            tensorboard_log=tensorboard_log_dir,
        )

        t0 = time.time()
        agent.train(
            total_timesteps=args.timesteps,
            log_interval=args.log_interval,
            save_path=model_path,
        )
        train_time = time.time() - t0

        print(f"\nEvaluating {name.upper()} ({args.eval_episodes} episodes)...")
        stats = evaluate_agent(
            agent,
            num_nodes=args.nodes,
            num_services=args.services,
            n_episodes=args.eval_episodes,
            seed=args.seed + 1000,
            eval_csv_path=eval_csv_path,
        )

        processed_rows = save_processed_monitor_csv(
            monitor_path=monitor_path,
            output_csv=processed_monitor_csv,
            smooth_window=args.smooth_window,
        )

        stats.update({
            "agent": name,
            "status": "OK",
            "train_time_sec": train_time,
            "model_path": model_path,
            "tensorboard_log_dir": tensorboard_log_dir,
            "monitor_csv": monitor_path,
            "processed_monitor_csv": processed_monitor_csv,
            "num_training_episodes": len(processed_rows),
        })

        print(
            f"Eval done | Mean reward: {stats['mean_reward']:.2f} "
            f"| Placed: {stats['mean_placed']*100:.1f}% "
            f"| Training episodes logged: {len(processed_rows)}"
        )
        return stats

    except Exception:
        print(f"\n[ERROR] [{name.upper()}] FAILED")
        traceback.print_exc()
        return {
            "agent": name,
            "status": "FAILED",
            "train_time_sec": 0.0,
            "mean_reward": "",
            "std_reward": "",
            "max_reward": "",
            "min_reward": "",
            "mean_placed": "",
            "model_path": model_path,
            "tensorboard_log_dir": tensorboard_log_dir,
            "monitor_csv": monitor_path,
            "processed_monitor_csv": processed_monitor_csv,
            "eval_episode_csv": eval_csv_path,
            "num_training_episodes": 0,
        }


# =============================================================================
# Summary
# =============================================================================

def save_training_summary(results: dict, args, paths: dict) -> str:
    summary_path = os.path.join(paths["experiment_dir"], "training_summary.csv")
    fieldnames = [
        "timestamp",
        "agent",
        "status",
        "num_nodes",
        "num_services",
        "timesteps",
        "seed",
        "eval_episodes",
        "train_time_sec",
        "mean_reward",
        "std_reward",
        "max_reward",
        "min_reward",
        "mean_placed",
        "num_training_episodes",
        "model_path",
        "tensorboard_log_dir",
        "monitor_csv",
        "processed_monitor_csv",
        "eval_episode_csv",
    ]

    rows = []
    for name, stats in results.items():
        rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "agent": name,
            "status": stats.get("status", ""),
            "num_nodes": args.nodes,
            "num_services": args.services,
            "timesteps": args.timesteps,
            "seed": args.seed,
            "eval_episodes": args.eval_episodes,
            "train_time_sec": stats.get("train_time_sec", ""),
            "mean_reward": stats.get("mean_reward", ""),
            "std_reward": stats.get("std_reward", ""),
            "max_reward": stats.get("max_reward", ""),
            "min_reward": stats.get("min_reward", ""),
            "mean_placed": stats.get("mean_placed", ""),
            "num_training_episodes": stats.get("num_training_episodes", ""),
            "model_path": stats.get("model_path", ""),
            "tensorboard_log_dir": stats.get("tensorboard_log_dir", ""),
            "monitor_csv": stats.get("monitor_csv", ""),
            "processed_monitor_csv": stats.get("processed_monitor_csv", ""),
            "eval_episode_csv": stats.get("eval_episode_csv", ""),
        })

    write_rows_csv(summary_path, rows, fieldnames)
    return summary_path


def print_summary(results: dict, args, paths: dict):
    print("\n\n" + "=" * 90)
    print("TRAINING SUMMARY")
    print(
        f"Config: {args.nodes} nodes | {args.services} services | "
        f"seed={args.seed} | {args.timesteps:,} timesteps"
    )
    print("=" * 90)
    print(
        f"{'Agent':<8} {'Mean±Std':>18} {'Max':>10} {'Min':>10} "
        f"{'Placed%':>9} {'Episodes':>10} {'Time':>10}  Status"
    )
    print("-" * 90)

    best_name = None
    best_reward = float("-inf")

    for name, stats in results.items():
        if stats.get("status") != "OK":
            print(
                f"{name.upper():<8} {'-':>18} {'-':>10} {'-':>10} "
                f"{'-':>9} {'-':>10} {'-':>10}  FAILED"
            )
            continue

        mean = float(stats["mean_reward"])
        std = float(stats["std_reward"])
        mx = float(stats["max_reward"])
        mn = float(stats["min_reward"])
        place = float(stats["mean_placed"]) * 100
        t = float(stats["train_time_sec"])
        episodes = int(stats.get("num_training_episodes", 0))

        mins, secs = divmod(int(t), 60)
        time_str = f"{mins}m{secs:02d}s"

        print(
            f"{name.upper():<8} {mean:>8.2f}±{std:<7.2f} "
            f"{mx:>10.2f} {mn:>10.2f} {place:>8.1f}% "
            f"{episodes:>10} {time_str:>10}  OK"
        )

        if mean > best_reward:
            best_reward = mean
            best_name = name

    print("=" * 90)
    if best_name:
        print(f"Best agent: {best_name.upper()} (mean reward = {best_reward:.2f})")

    print("\nArtifacts:")
    print(f"  Experiment dir       : {paths['experiment_dir']}")
    print(f"  Models               : {paths['models_dir']}")
    print(f"  TensorBoard logs     : {paths['tensorboard_dir']}")
    print(f"  Monitor CSVs         : {paths['monitor_dir']}")
    print(f"  Evaluation CSVs      : {paths['eval_dir']}")
    print(f"  Training curves      : {paths['figures_dir']}")
    print(f"  Summary CSV          : {os.path.join(paths['experiment_dir'], 'training_summary.csv')}")
    print("\nTensorBoard:")
    print(f"  tensorboard --logdir {paths['tensorboard_dir']}")
    print("=" * 90 + "\n")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and compare DRL agents for K8s Microservice Placement"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=["ppo", "dqn", "a2c"],
        default=["ppo", "dqn", "a2c"],
        help="Agents to train",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000,
        help="Timesteps per agent",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=5,
        help="Number of nodes",
    )
    parser.add_argument(
        "--services",
        type=int,
        default=8,
        help="Number of services",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Evaluation episodes after training",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Terminal log interval used by agent.train",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=100,
        help="Moving average window for training curves",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Output folder for this training run",
    )
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    args = parse_args()

    if args.experiment_dir is None:
        args.experiment_dir = default_experiment_dir(args)

    paths = {
        "experiment_dir": ensure_dir(args.experiment_dir),
        "models_dir": ensure_dir(os.path.join(args.experiment_dir, "models")),
        "tensorboard_dir": ensure_dir(os.path.join(args.experiment_dir, "tensorboard")),
        "monitor_dir": ensure_dir(os.path.join(args.experiment_dir, "monitor")),
        "figures_dir": ensure_dir(os.path.join(args.experiment_dir, "training_curves")),
        "eval_dir": ensure_dir(os.path.join(args.experiment_dir, "eval")),
    }

    write_json(
        os.path.join(paths["experiment_dir"], "experiment_config.json"),
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "agents": args.agents,
            "timesteps": args.timesteps,
            "seed": args.seed,
            "nodes": args.nodes,
            "services": args.services,
            "eval_episodes": args.eval_episodes,
            "log_interval": args.log_interval,
            "smooth_window": args.smooth_window,
            "experiment_dir": args.experiment_dir,
        },
    )

    print("\n" + "=" * 70)
    print("K8s MICROSERVICE PLACEMENT - DRL TRAINING")
    print("=" * 70)
    print(f"Agents         : {[a.upper() for a in args.agents]}")
    print(f"Timesteps      : {args.timesteps:,} per agent")
    print(f"Seed           : {args.seed}")
    print(f"Cluster        : {args.nodes} nodes | {args.services} services")
    print(f"Experiment dir : {paths['experiment_dir']}")
    print("=" * 70)

    results = {}
    total_t0 = time.time()

    for name in args.agents:
        results[name] = train_one(name, args, paths)

    total_time = time.time() - total_t0

    processed_by_agent = {}
    for name, stats in results.items():
        processed_csv = stats.get("processed_monitor_csv", "")
        # Plot from processed rows if available; otherwise parse monitor again.
        rows = []
        if processed_csv and os.path.exists(processed_csv):
            with open(processed_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        rows.append({
                            "episode": int(row["episode"]),
                            "reward": float(row["reward"]),
                            "rolling_reward": float(row["rolling_reward"]),
                            "length": int(float(row["length"])),
                            "time": float(row["time"]),
                        })
                    except Exception:
                        continue
        processed_by_agent[name] = rows

    make_learning_curve_plots(
        processed_by_agent=processed_by_agent,
        figures_dir=paths["figures_dir"],
        smooth_window=args.smooth_window,
    )

    summary_path = save_training_summary(results, args, paths)

    mins, secs = divmod(int(total_time), 60)
    print(f"\nTotal training time: {mins}m{secs:02d}s")
    print_summary(results, args, paths)
    print(f"Saved summary: {summary_path}")

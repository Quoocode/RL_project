import os
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def load_episode_csv(path):
    data = defaultdict(list)
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            r2 = {k: float(v) if v.replace('.', '', 1).replace('-', '', 1).isdigit() else v for k, v in r.items()}
            rows.append(r2)
            agent = r['agent']
            data[agent].append(r2)
    return data, rows


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _find_event_files(log_dir: str) -> List[str]:
    event_files: List[str] = []
    for root, _, files in os.walk(log_dir):
        for fn in files:
            if fn.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, fn))
    event_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return event_files


def _load_scalars_from_event_file(event_file: str) -> Dict[str, List[Tuple[int, float]]]:
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    scalars: Dict[str, List[Tuple[int, float]]] = {}

    for tag in tags:
        events = ea.Scalars(tag)
        points = [(int(e.step), float(e.value)) for e in events]
        if points:
            scalars[tag] = points

    return scalars


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, kernel, mode="same")


def _plot_scalar_series(series_map: Dict[str, Dict[str, List[Tuple[int, float]]]],
                        tag: str,
                        out_path: str,
                        title: str,
                        y_label: str,
                        smooth_window: int = 1):
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0

    for agent, scalars in series_map.items():
        if tag not in scalars:
            continue
        points = sorted(scalars[tag], key=lambda t: t[0])
        x = np.array([p[0] for p in points], dtype=float)
        y = np.array([p[1] for p in points], dtype=float)

        if smooth_window > 1:
            y_plot = _smooth(y, smooth_window)
            ax.plot(x, y_plot, label=f"{agent} (smoothed)", linewidth=2)
            ax.plot(x, y, alpha=0.18, linewidth=1)
        else:
            ax.plot(x, y, label=agent, linewidth=2)

        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print(f"  ⚠️  Skip {os.path.basename(out_path)} (tag not found: {tag})")
        return

    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"  ✅ Saved {out_path}")


def _plot_agent_losses(agent: str,
                       scalars: Dict[str, List[Tuple[int, float]]],
                       outdir: str,
                       smooth_window: int = 1):
    # Common SB3 loss-like tags across PPO/A2C/DQN.
    candidate_tags = [
        "train/loss",
        "train/value_loss",
        "train/policy_gradient_loss",
        "train/policy_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/explained_variance",
    ]
    present = [t for t in candidate_tags if t in scalars]
    if not present:
        print(f"  ⚠️  Skip loss chart for {agent} (no loss-like tags)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for tag in present:
        points = sorted(scalars[tag], key=lambda t: t[0])
        x = np.array([p[0] for p in points], dtype=float)
        y = np.array([p[1] for p in points], dtype=float)
        if smooth_window > 1:
            y = _smooth(y, smooth_window)
        ax.plot(x, y, label=tag)

    ax.set_title(f"Training losses/diagnostics — {agent}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(outdir, f"losses_{agent}.png")
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"  ✅ Saved {out_path}")


def analyze_training_logs(log_root: str,
                          agents: List[str],
                          outdir: str,
                          smooth_window: int = 1):
    os.makedirs(outdir, exist_ok=True)

    series_map: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}

    for agent in agents:
        agent_log_dir = os.path.join(log_root, agent)
        if not os.path.isdir(agent_log_dir):
            print(f"  ⚠️  Log dir not found for {agent}: {agent_log_dir}")
            continue

        event_files = _find_event_files(agent_log_dir)
        if not event_files:
            print(f"  ⚠️  No TensorBoard event file for {agent} in {agent_log_dir}")
            continue

        latest_event = event_files[0]
        print(f"  ℹ️  Use latest event file for {agent}: {latest_event}")
        scalars = _load_scalars_from_event_file(latest_event)
        series_map[agent.upper()] = scalars

    if not series_map:
        print("No training logs loaded. Nothing to plot.")
        return

    # Convergence charts (cross-agent)
    _plot_scalar_series(
        series_map,
        tag="rollout/ep_rew_mean",
        out_path=os.path.join(outdir, "convergence_reward.png"),
        title="Convergence — Mean Episode Reward",
        y_label="Episode reward mean",
        smooth_window=smooth_window,
    )

    _plot_scalar_series(
        series_map,
        tag="rollout/ep_len_mean",
        out_path=os.path.join(outdir, "convergence_episode_length.png"),
        title="Convergence — Mean Episode Length",
        y_label="Episode length mean",
        smooth_window=smooth_window,
    )

    # Optional per-agent loss diagnostics
    for agent, scalars in series_map.items():
        _plot_agent_losses(agent, scalars, outdir, smooth_window=smooth_window)

    print(f"\n  ✅ Training plots exported to: {outdir}")


def boxplot_rewards(data, outdir):
    agents = sorted(data.keys())
    rewards = [ [d['reward'] for d in data[a]] for a in agents ]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.boxplot(rewards, labels=agents, showfliers=True)
    ax.set_title('Reward distribution per agent')
    ax.set_ylabel('Episode reward')
    plt.xticks(rotation=20)
    plt.tight_layout()
    path = os.path.join(outdir, 'reward_boxplot.png')
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  ✅ Saved {path}")


def hist_rewards(data, outdir):
    os.makedirs(outdir, exist_ok=True)
    for agent, rows in data.items():
        vals = [r['reward'] for r in rows]
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(vals, bins=20, alpha=0.8)
        ax.set_title(f'Reward histogram — {agent}')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Count')
        plt.tight_layout()
        path = os.path.join(outdir, f'hist_reward_{agent}.png')
        plt.savefig(path, dpi=200)
        plt.close(fig)
        print(f"  ✅ Saved {path}")


def scatter_latency_vs_reward(rows, outdir):
    agents = sorted(set(r['agent'] for r in rows))
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10,6))
    for agent in agents:
        rs = [r for r in rows if r['agent'] == agent]
        x = [r.get('avg_effective_latency_cost', r.get('effective_latency_cost', 0.0)) for r in rs]
        y = [r['reward'] for r in rs]
        ax.scatter(x, y, label=agent, alpha=0.7)
    ax.set_xlabel('Avg effective latency cost')
    ax.set_ylabel('Reward')
    ax.set_title('Latency vs Reward (per episode)')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(outdir, 'latency_vs_reward.png')
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  ✅ Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['evaluation', 'training'], default='evaluation')

    # Evaluation CSV mode
    parser.add_argument('--input', default='./results/evaluation_episode_metrics.csv')

    # Training TensorBoard mode
    parser.add_argument('--log-root', default='./logs',
                        help='Root log directory that contains agent folders (ppo/dqn/a2c).')
    parser.add_argument('--agents', nargs='+', default=['ppo', 'dqn', 'a2c'])
    parser.add_argument('--smooth-window', type=int, default=1,
                        help='Moving average window for training plots (default: 1 = no smoothing).')

    parser.add_argument('--outdir', default='./results/analysis_plots')
    args = parser.parse_args()

    if args.mode == 'evaluation':
        if not os.path.exists(args.input):
            print(f'Input not found: {args.input}')
            return

        os.makedirs(args.outdir, exist_ok=True)
        data, rows = load_episode_csv(args.input)

        boxplot_rewards(data, args.outdir)
        hist_rewards(data, args.outdir)
        scatter_latency_vs_reward(rows, args.outdir)
    else:
        analyze_training_logs(
            log_root=args.log_root,
            agents=[a.lower() for a in args.agents],
            outdir=args.outdir,
            smooth_window=max(1, int(args.smooth_window)),
        )


if __name__ == '__main__':
    main()

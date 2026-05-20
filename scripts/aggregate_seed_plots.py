import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def find_event_files(path):
    files = []
    for root, _, filenames in os.walk(path):
        for fn in filenames:
            if fn.startswith("events.out.tfevents"):
                files.append(os.path.join(root, fn))
    files.sort()
    return files


def load_series(event_file, tag="rollout/ep_rew_mean"):
    ea = event_accumulator.EventAccumulator(event_file, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        return []
    events = ea.Scalars(tag)
    return [(int(e.step), float(e.value)) for e in events]


def interp_series(points, x_common):
    if not points:
        return np.full_like(x_common, np.nan, dtype=float)
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    # if single point, repeat
    if x.size == 1:
        return np.full_like(x_common, y[0], dtype=float)
    # remove duplicate x
    ux, idx = np.unique(x, return_index=True)
    uy = y[idx]
    return np.interp(x_common, ux, uy)


def aggregate(agent_log_root: str, seeds: list[int], outdir: str, smooth: int = 25):
    os.makedirs(outdir, exist_ok=True)
    all_series = []
    for s in seeds:
        seed_dir = os.path.join(agent_log_root, f"seed_{s}")
        if not os.path.isdir(seed_dir):
            # fallback: look under agent_log_root for event files containing seed
            seed_dir = agent_log_root
        ev_files = find_event_files(seed_dir)
        if not ev_files:
            print(f"  ⚠️  No event file for seed {s} in {seed_dir}")
            continue
        # use latest
        ev = ev_files[-1]
        pts = load_series(ev, tag="rollout/ep_rew_mean")
        if not pts:
            print(f"  ⚠️  No rollout/ep_rew_mean tag in {ev}")
            continue
        all_series.append(pts)
        print(f"  ℹ️  Loaded seed {s} series, {len(pts)} points")

    if not all_series:
        print("No series loaded. Exiting.")
        return

    # determine common x (union, sorted)
    xs = sorted({int(x) for s in all_series for x, _ in s})
    # downsample common x if too large
    max_points = 2000
    if len(xs) > max_points:
        xs = np.linspace(min(xs), max(xs), max_points, dtype=int)
    x_common = np.array(xs, dtype=float)

    mat = np.vstack([interp_series(s, x_common) for s in all_series])
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)

    # smoothing
    def smooth_arr(a, w):
        if w <= 1:
            return a
        kernel = np.ones(w) / w
        return np.convolve(a, kernel, mode='same')

    mean_s = smooth_arr(mean, smooth)
    std_s = smooth_arr(std, smooth)

    # plot mean ± std
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(x_common, mean_s, label='Mean reward', color='#2c3e50')
    ax.fill_between(x_common, mean_s - std_s, mean_s + std_s, color='#7f8c8d', alpha=0.3, label='Std')
    ax.set_title('DQN — Mean Episode Reward across seeds')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Episode reward')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend()
    out1 = os.path.join(outdir, 'dqn_mean_reward_across_seeds.png')
    plt.tight_layout()
    plt.savefig(out1, dpi=220)
    plt.close(fig)
    print(f"  ✅ Saved {out1}")

    # also plot raw per-seed lines
    fig, ax = plt.subplots(figsize=(10,6))
    for i, s in enumerate(all_series):
        y = interp_series(s, x_common)
        ax.plot(x_common, y, alpha=0.6, label=f'seed_{seeds[i]}')
    ax.set_title('DQN — Per-seed reward curves')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Episode reward')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend()
    out2 = os.path.join(outdir, 'dqn_per_seed_reward_curves.png')
    plt.tight_layout()
    plt.savefig(out2, dpi=220)
    plt.close(fig)
    print(f"  ✅ Saved {out2}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-root', default='./logs/dqn')
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--outdir', default='./results/training_plots/dqn_multi_seed')
    parser.add_argument('--smooth', type=int, default=25)
    args = parser.parse_args()
    aggregate(args.log_root, args.seeds, args.outdir, smooth=args.smooth)

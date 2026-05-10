# agents/ppo_agent.py
"""
PPO Agent cho bài toán Microservice Placement trên Kubernetes.

Chạy nhanh (mặc định):
    python agents/ppo_agent.py

Tuỳ chỉnh qua CLI:
    python agents/ppo_agent.py --timesteps 200000 --seed 42
    python agents/ppo_agent.py --timesteps 50000 --nodes 8 --services 10
    python agents/ppo_agent.py --resume ./models/ppo_model  (tiếp tục train)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from envs.k8s_env import K8sPlacementEnv


# ═════════════════════════════════════════════════════════════════════════════
# SEED — gọi trước khi tạo bất cứ thứ gì
# ═════════════════════════════════════════════════════════════════════════════
def set_global_seed(seed: int):
    """Seed tất cả nguồn randomness để kết quả reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    # torch seed sẽ được truyền qua PPO(seed=seed) bên dưới


# ═════════════════════════════════════════════════════════════════════════════
# CALLBACK
# ═════════════════════════════════════════════════════════════════════════════
class PlacementCallback(BaseCallback):
    """
    Theo dõi quá trình học và log thêm metrics vào Tensorboard.
    Monitor wrapper đã tự log ep_rew_mean và ep_len_mean,
    callback này bổ sung log mỗi N episode để dễ đọc trên terminal.
    """

    def __init__(self, log_interval: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) == 0:
            return True

        ep_info = self.model.ep_info_buffer[-1]

        # ep_info['r'] là episode reward do Monitor ghi lại
        self.episode_rewards.append(ep_info['r'])
        n_ep = len(self.episode_rewards)

        # Log thêm vào Tensorboard
        self.logger.record("train/episode_count", n_ep)

        # In terminal mỗi log_interval episode
        if self.verbose > 0 and n_ep % self.log_interval == 0:
            recent = self.episode_rewards[-self.log_interval:]
            mean_r = np.mean(recent)
            max_r  = np.max(recent)
            min_r  = np.min(recent)
            print(
                f"  Episode {n_ep:>6} | "
                f"Mean: {mean_r:>7.2f} | "
                f"Max: {max_r:>7.2f} | "
                f"Min: {min_r:>7.2f} | "
                f"Timestep: {self.num_timesteps:>8}"
            )

        return True


# ═════════════════════════════════════════════════════════════════════════════
# PPO AGENT
# ═════════════════════════════════════════════════════════════════════════════
class PPOAgent:
    """Quản lý vòng đời của PPO model: khởi tạo, train, save, load, predict."""

    def __init__(self, env, seed: int = 42, tensorboard_log: str = "./logs/ppo"):
        self.env = env
        self.seed = seed

        self.model = PPO(
            policy          = "MlpPolicy",
            env             = env,
            learning_rate   = 3e-4,
            n_steps         = 256,
            batch_size      = 64,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.01,
            verbose         = 0,
            seed            = seed,         # ← seed cho SB3 / torch
            tensorboard_log = tensorboard_log,
        )

    @classmethod
    def load(cls, path: str, env, seed: int = 42,
             tensorboard_log: str = "./logs/ppo"):
        """
        Load model đã lưu để tiếp tục train hoặc evaluate.
        Dùng khi resume training bị ngắt giữa chừng.
        """
        agent = cls.__new__(cls)
        agent.env  = env
        agent.seed = seed
        agent.model = PPO.load(
            path,
            env             = env,
            tensorboard_log = tensorboard_log,
        )
        print(f"  ✅ Đã load model từ: {path}")
        return agent

    def train(self, total_timesteps: int = 100_000,
              log_interval: int = 100,
              save_path: str = "./models/ppo_model"):
        """
        Huấn luyện model.
        - total_timesteps : tổng số bước môi trường
        - log_interval    : in terminal mỗi N episode
        - save_path       : đường dẫn lưu model (không cần .zip)
        """
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        print(f"\n{'='*55}")
        print(f"  PPO TRAINING")
        print(f"{'='*55}")
        print(f"  Seed          : {self.seed}")
        print(f"  Timesteps     : {total_timesteps:,}")
        print(f"  Save path     : {save_path}.zip")
        print(f"  Tensorboard   : {self.model.tensorboard_log}")
        print(f"  Nodes/Services: {self.env.unwrapped.num_nodes} / "
              f"{self.env.unwrapped.num_services}")
        print(f"{'='*55}\n")

        callback = PlacementCallback(log_interval=log_interval, verbose=1)

        self.model.learn(
            total_timesteps  = total_timesteps,
            callback         = callback,
            progress_bar     = True,
            reset_num_timesteps = False,   # Giữ timestep count khi resume
        )

        self.model.save(save_path)
        print(f"\n  ✅ Đã lưu model tại: {save_path}.zip")

    def predict(self, observation, deterministic: bool = True):
        """
        Dự đoán hành động từ observation.
        deterministic=True  : luôn chọn action tốt nhất (dùng khi evaluate)
        deterministic=False : có random (dùng khi collect experience)
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action


# ═════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO agent cho K8s Microservice Placement"
    )
    parser.add_argument("--timesteps", type=int,   default=100_000,
                        help="Tổng số timestep để train (default: 100000)")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--nodes",     type=int,   default=5,
                        help="Số node trong cluster (default: 5)")
    parser.add_argument("--services",  type=int,   default=5,
                        help="Số microservice trong chain (default: 5)")
    parser.add_argument("--save",      type=str,   default="./models/ppo_model",
                        help="Đường dẫn lưu model (default: ./models/ppo_model)")
    parser.add_argument("--resume",    type=str,   default=None,
                        help="Load model cũ để train tiếp (vd: ./models/ppo_model)")
    parser.add_argument("--log-dir",   type=str,   default="./logs/ppo",
                        help="Thư mục Tensorboard log (default: ./logs/ppo)")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="In terminal mỗi N episode (default: 100)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 1. Set seed toàn cục TRƯỚC KHI tạo bất cứ thứ gì
    set_global_seed(args.seed)

    # 2. Tạo môi trường, bọc Monitor để Tensorboard có đủ dữ liệu
    env = K8sPlacementEnv(
        num_nodes    = args.nodes,
        num_services = args.services,
    )
    env = Monitor(env, filename=None)  # filename=None → không ghi file .csv
    env.reset(seed=args.seed)

    # 3. Tạo hoặc load agent
    if args.resume:
        print(f"\n  Đang load model từ {args.resume} để train tiếp...")
        agent = PPOAgent.load(args.resume, env,
                              seed=args.seed,
                              tensorboard_log=args.log_dir)
    else:
        agent = PPOAgent(env, seed=args.seed, tensorboard_log=args.log_dir)

    # 4. Train
    agent.train(
        total_timesteps = args.timesteps,
        log_interval    = args.log_interval,
        save_path       = args.save,
    )

    # 5. Hướng dẫn xem Tensorboard
    print(f"\n  Xem kết quả training:")
    print(f"  tensorboard --logdir {args.log_dir}")

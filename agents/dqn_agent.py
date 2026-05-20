# agents/dqn_agent.py
"""
DQN Agent cho bài toán Microservice Placement trên Kubernetes.

Chạy nhanh (mặc định):
    python agents/dqn_agent.py

Tuỳ chỉnh qua CLI:
    python agents/dqn_agent.py --timesteps 200000 --seed 42
    python agents/dqn_agent.py --timesteps 50000 --nodes 8 --services 10
    python agents/dqn_agent.py --resume ./models/dqn_model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# Tái sử dụng PlacementCallback và set_global_seed từ ppo_agent
from agents.ppo_agent import PlacementCallback, set_global_seed

from envs.k8s_env import K8sPlacementEnv
from envs.k8s_env import K8sPlacementEnvDynamic


class DQNAgent:
    """Quản lý vòng đời của DQN model: khởi tạo, train, save, load, predict."""

    def __init__(self, env, seed: int = 42, tensorboard_log: str = "./logs/dqn"):
        self.env  = env
        self.seed = seed

        # Disable tensorboard logging if tensorboard not available
        try:
            import tensorboard  # type: ignore
            tb_log = tensorboard_log
        except Exception:
            tb_log = None

        self.model = DQN(
            policy                 = "MlpPolicy",
            env                    = env,
            learning_rate          = 1e-3,
            buffer_size            = 100_000,
            learning_starts        = 1_000,
            batch_size             = 64,
            gamma                  = 0.99,
            target_update_interval = 250,
            exploration_fraction   = 0.1,   # 10% đầu khám phá ngẫu nhiên
            exploration_final_eps  = 0.05,  # Sau đó giữ 5% random
            verbose                = 0,
            seed                   = seed,
            tensorboard_log        = tb_log,
        )

    @classmethod
    def load(cls, path: str, env, seed: int = 42,
             tensorboard_log: str = "./logs/dqn"):
        """Load model đã lưu để tiếp tục train hoặc evaluate."""
        agent = cls.__new__(cls)
        agent.env  = env
        agent.seed = seed
        agent.model = DQN.load(
            path,
            env             = env,
            tensorboard_log = tensorboard_log,
        )
        print(f"  ✅ Đã load model từ: {path}")
        return agent

    def train(self, total_timesteps: int = 100_000,
              log_interval: int = 100,
              save_path: str = "./models/dqn_model"):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        print(f"\n{'='*55}")
        print(f"  DQN TRAINING")
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
            total_timesteps     = total_timesteps,
            callback            = callback,
            progress_bar        = True,
            reset_num_timesteps = False,
        )

        self.model.save(save_path)
        print(f"\n  ✅ Đã lưu model tại: {save_path}.zip")

    def predict(self, observation, deterministic: bool = True):
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action


# ═════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DQN agent cho K8s Microservice Placement"
    )
    parser.add_argument("--timesteps",    type=int, default=100_000)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--nodes",        type=int, default=5)
    parser.add_argument("--services",     type=int, default=5)
    parser.add_argument("--max-nodes",    type=int, default=None)
    parser.add_argument("--max-services", type=int, default=None)
    parser.add_argument("--save",         type=str, default="./models/dqn_model")
    parser.add_argument("--resume",       type=str, default=None)
    parser.add_argument("--log-dir",      type=str, default="./logs/dqn")
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    set_global_seed(args.seed)

    if args.max_nodes is not None or args.max_services is not None:
        env = K8sPlacementEnvDynamic(
            actual_num_nodes=args.nodes,
            actual_num_services=args.services,
            max_nodes=args.max_nodes or args.nodes,
            max_services=args.max_services or args.services,
        )
    else:
        env = K8sPlacementEnv(num_nodes=args.nodes, num_services=args.services)
    env = Monitor(env, filename=None)
    env.reset(seed=args.seed)

    if args.resume:
        print(f"\n  Đang load model từ {args.resume} để train tiếp...")
        agent = DQNAgent.load(args.resume, env,
                              seed=args.seed,
                              tensorboard_log=args.log_dir)
    else:
        agent = DQNAgent(env, seed=args.seed, tensorboard_log=args.log_dir)

    agent.train(
        total_timesteps = args.timesteps,
        log_interval    = args.log_interval,
        save_path       = args.save,
    )

    print(f"\n  Xem kết quả training:")
    print(f"  tensorboard --logdir {args.log_dir}")

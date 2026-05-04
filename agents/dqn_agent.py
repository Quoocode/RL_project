# agents/dqn_agent.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from agents.ppo_agent import PlacementCallback # Tái sử dụng callback để có progress bar
import numpy as np

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.model = DQN(
            "MlpPolicy", 
            env,
            learning_rate=1e-3, # DQN thường cần learning rate cao hơn PPO một chút
            buffer_size=100000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            target_update_interval=250,
            verbose=0,
            tensorboard_log="./logs/dqn/"
        )
    
    def train(self, total_timesteps: int = 100000):
        print(f"BẮT ĐẦU HUẤN LUYỆN DQN ({total_timesteps} BƯỚC)...")
        callback = PlacementCallback(verbose=1)
        self.model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
        print("Huấn luyện DQN hoàn tất!")
    
    def predict(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def save(self, path: str):
        self.model.save(path)

if __name__ == "__main__":
    from envs.k8s_env import K8sPlacementEnv
    env = K8sPlacementEnv(num_nodes=5, num_services=30) # Kịch bản khó
    agent = DQNAgent(env)
    agent.train(total_timesteps=100000)
    
    os.makedirs("./models", exist_ok=True)
    agent.save("./models/dqn_hard_model")
    print("Đã lưu bộ não DQN tại ./models/dqn_hard_model.zip")
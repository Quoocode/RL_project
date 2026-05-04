# agents/a2c_agent.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import A2C
from agents.ppo_agent import PlacementCallback
import numpy as np

class A2CAgent:
    def __init__(self, env):
        self.env = env
        self.model = A2C(
            "MlpPolicy", 
            env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            verbose=0,
            tensorboard_log="./logs/a2c/"
        )
    
    def train(self, total_timesteps: int = 100000):
        print(f"BẮT ĐẦU HUẤN LUYỆN A2C ({total_timesteps} BƯỚC)...")
        callback = PlacementCallback(verbose=1)
        self.model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
        print("Huấn luyện A2C hoàn tất!")
    
    def predict(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def save(self, path: str):
        self.model.save(path)

if __name__ == "__main__":
    from envs.k8s_env import K8sPlacementEnv
    env = K8sPlacementEnv(num_nodes=5, num_services=30) # Kịch bản khó
    agent = A2CAgent(env)
    agent.train(total_timesteps=100000)
    
    os.makedirs("./models", exist_ok=True)
    agent.save("./models/a2c_hard_model")
    print("Đã lưu bộ não A2C tại ./models/a2c_hard_model.zip")
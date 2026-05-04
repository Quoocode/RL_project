# agents/ppo_agent.py
import sys
import os

# Ép Python lùi ra thư mục gốc để không bị lỗi không tìm thấy module 'envs'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class PlacementCallback(BaseCallback):
    """Callback để theo dõi quá trình AI học."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
    
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(ep_info['r'])
            
            # Cứ 100 ván in ra điểm 1 lần để xem AI có thông minh lên không
            if self.verbose > 0 and len(self.episode_rewards) % 100 == 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                print(f"Đã chơi {len(self.episode_rewards)} ván | Điểm trung bình 100 ván gần nhất: {mean_reward:.2f}")
        return True

class PPOAgent:
    """Class quản lý thuật toán PPO."""
    def __init__(self, env):
        self.env = env
        self.model = PPO(
            "MlpPolicy", 
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            verbose=0,
            tensorboard_log="./logs/"
        )
    
    def train(self, total_timesteps: int = 10000):
        print(f"BẮT ĐẦU HUẤN LUYỆN AI ({total_timesteps} BƯỚC)...")
        callback = PlacementCallback(verbose=1)
        self.model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
        print("Huấn luyện hoàn tất!")
    
    def predict(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def save(self, path: str):
        self.model.save(path)
        
# ==================== TEST TRAINING ====================
if __name__ == "__main__":
    from envs.k8s_env import K8sPlacementEnv
    
    # Cập nhật môi trường huấn luyện thành kịch bản khó (30 Services)
    env = K8sPlacementEnv(num_nodes=5, num_services=30)
    agent = PPOAgent(env)
    
    # Train 100,000 bước
    agent.train(total_timesteps=100000)
    
    os.makedirs("./models", exist_ok=True)
    # Lưu với tên khác để không ghi đè model cũ
    agent.save("./models/ppo_hard_model") 
    print("Đã lưu bộ não AI tại ./models/ppo_hard_model.zip")
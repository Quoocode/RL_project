# evaluate.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt # Import thư viện vẽ biểu đồ
from envs.k8s_env import K8sPlacementEnv
from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent
from agents.a2c_agent import A2CAgent

def run_random_agent(env, n_episodes=10):
    print("\n--- ĐANG CHẠY THUẬT TOÁN NGẪU NHIÊN (RANDOM) ---")
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    mean_reward = np.mean(rewards)
    print(f"Điểm trung bình (Random): {mean_reward:.2f}")
    return mean_reward

def run_first_fit_agent(env, n_episodes=10):
    print("\n--- ĐANG CHẠY THUẬT TOÁN FIRST-FIT ---")
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = 0
            service = env.service_chain.services[env.current_service_idx]
            for i, node in enumerate(env.topology.nodes):
                if node.can_allocate(service.cpu_request, service.memory_request):
                    action = i
                    break
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    mean_reward = np.mean(rewards)
    print(f"Điểm trung bình (First-Fit): {mean_reward:.2f}")
    return mean_reward

def run_ai_agent(env, AgentClass, model_path, agent_name, n_episodes=10):
    print(f"\n--- ĐANG CHẠY {agent_name} ---")
    agent = AgentClass(env)
    agent.model = agent.model.load(model_path)
    rewards = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0 # Thêm biến đếm số bước
        
        # Bẫy vòng lặp: Nếu quá 100 bước (cho 30 services) mà chưa xong thì ép dừng
        while not done and step_count < 100:
            # Nếu bị kẹt quá 50 bước ở cùng 1 service, cho AI chọn đại 1 Node ngẫu nhiên
            if step_count > 50:
                action = env.action_space.sample() 
            else:
                action = agent.predict(obs)
                
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
        if step_count >= 100:
            # Phạt thêm điểm nếu AI quá ngoan cố
            episode_reward -= 100 
            print(f"   ⚠️ {agent_name} bị kẹt vòng lặp lặp vô tận. Bị giám thị thu bài!")
            
        rewards.append(episode_reward)
        
    mean_reward = np.mean(rewards)
    print(f"Điểm trung bình ({agent_name}): {mean_reward:.2f}")
    return mean_reward

def draw_dynamic_chart(methods, scores):
    """Hàm vẽ biểu đồ tự động dựa trên danh sách điểm số thực tế."""
    # 5 màu sắc đẹp mắt cho 5 thuật toán
    colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#2ecc71'] 
    
    plt.figure(figsize=(12, 6)) # Mở rộng bề ngang để chứa tối đa 5 cột
    
    # Cắt bớt màu nếu số lượng thuật toán chạy thành công ít hơn 5
    active_colors = colors[:len(methods)]
    
    bars = plt.bar(methods, scores, color=active_colors, width=0.6)
    
    # Hiển thị điểm số trên đỉnh mỗi cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, 
                 f'{yval:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('So sánh Hiệu suất Sắp xếp Microservices (Kịch bản 30 Services)', fontsize=16, pad=20)
    plt.ylabel('Điểm thưởng trung bình (Reward)', fontsize=12)
    
    # Tự động co giãn trục Y để biểu đồ luôn đẹp, không bị lố chiều cao
    min_score = min(scores) if min(scores) < 0 else 0
    plt.ylim(min_score - 20, max(scores) + 30) 
    plt.axhline(0, color='black', linewidth=1) # Vẽ thêm một đường kẻ ngang ở mức 0
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs("./results", exist_ok=True)
    save_path = "./results/comparison_chart.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Đã xuất biểu đồ thành công tại: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    print("=== BẮT ĐẦU CHẠY THỰC NGHIỆM ĐA MÔ HÌNH ===")
    env_hard = K8sPlacementEnv(num_nodes=5, num_services=30)
    
    # Hai mảng trống để hứng tên thuật toán và điểm số
    methods = []
    scores = []
    
    # 1. Thuật toán Random
    scores.append(run_random_agent(env_hard))
    methods.append('Ngẫu nhiên\n(Random)')
    
    # 2. Thuật toán First-Fit
    scores.append(run_first_fit_agent(env_hard))
    methods.append('Truyền thống\n(First-Fit)')
    
    # 3. AI DQN
    try:
        scores.append(run_ai_agent(env_hard, DQNAgent, "./models/dqn_hard_model", "AI DQN"))
        methods.append('AI\n(DQN)')
    except Exception as e:
        print(f"⚠️ Bỏ qua DQN vì chưa tìm thấy file model đã huấn luyện.")

    # 4. AI A2C
    try:
        scores.append(run_ai_agent(env_hard, A2CAgent, "./models/a2c_hard_model", "AI A2C"))
        methods.append('AI\n(A2C)')
    except Exception as e:
        print(f"⚠️ Bỏ qua A2C vì chưa tìm thấy file model đã huấn luyện.")

    # 5. AI PPO
    try:
        scores.append(run_ai_agent(env_hard, PPOAgent, "./models/ppo_hard_model", "AI PPO"))
        methods.append('AI\n(PPO)')
    except Exception as e:
        print(f"⚠️ Bỏ qua PPO vì chưa tìm thấy file model đã huấn luyện.")
    
    print("\n=== HOÀN TẤT THỰC NGHIỆM ===")
    
    # Tự động gọi hàm vẽ biểu đồ sau khi chạy xong tất cả
    if len(methods) > 0:
        draw_dynamic_chart(methods, scores)
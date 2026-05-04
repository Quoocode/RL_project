# run_all.py
import subprocess
import time
import os

def run_script(script_path):
    if not os.path.exists(script_path):
        print(f"❌ Lỗi: Không tìm thấy file {script_path}")
        return
    
    print(f"\n" + "="*50)
    print(f">>> ĐANG CHẠY: {script_path} ...")
    print("="*50)
    
    start_time = time.time()
    # Chạy script và đợi nó kết thúc
    process = subprocess.Popen(["python", script_path])
    process.wait()
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n✅ HOÀN THÀNH {script_path} TRONG {duration:.2f} GIÂY (~{duration/60:.2f} PHÚT).")

if __name__ == "__main__":
    print("🚀 HỆ THỐNG HUẤN LUYỆN TỰ ĐỘNG TOÀN DIỆN (DQN, A2C, PPO)")
    print("Ghi chú: Quá trình này có thể mất từ 5-10 phút tùy cấu hình máy.\n")
    
    # Danh sách các script huấn luyện
    training_scripts = [
        "agents/dqn_agent.py",
        "agents/a2c_agent.py",
        "agents/ppo_agent.py"
    ]
    
    # 1. Chạy huấn luyện lần lượt các AI
    for script in training_scripts:
        run_script(script)
    
    # 2. Đánh giá và xuất biểu đồ so sánh cuối cùng
    print("\n" + "#"*60)
    print("### BẮT ĐẦU ĐÁNH GIÁ VÀ XUẤT BIỂU ĐỒ TỔNG HỢP ###")
    print("#"*60)
    run_script("evaluate.py")
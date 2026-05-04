# plot_results.py
import matplotlib.pyplot as plt
import numpy as np
import os

def draw_comparison_chart():
    # Dữ liệu thực tế từ terminal của bạn
    methods = ['Ngẫu nhiên\n(Random)', 'Truyền thống\n(First-Fit)', 'Trí tuệ nhân tạo\n(AI - PPO)']
    scores = [97.74, 137.33, 138.62]
    
    # Setup màu sắc
    colors = ['#e74c3c', '#f39c12', '#2ecc71'] # Đỏ, Vàng, Xanh lá
    
    # Tạo biểu đồ
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, scores, color=colors, width=0.6)
    
    # Thêm số điểm lên trên từng cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, 
                 f'{yval:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Trang trí biểu đồ
    plt.title('So sánh Hiệu suất Sắp xếp Microservices (Kịch bản 30 Services)', fontsize=16, pad=20)
    plt.ylabel('Điểm thưởng trung bình (Reward)', fontsize=12)
    plt.ylim(0, 160) # Chỉnh trục Y cho đẹp
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Lưu hình ảnh
    os.makedirs("./results", exist_ok=True)
    save_path = "./results/comparison_chart.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã vẽ xong! Biểu đồ được lưu tại: {save_path}")
    
    # Hiển thị lên màn hình
    plt.show()

if __name__ == "__main__":
    draw_comparison_chart()
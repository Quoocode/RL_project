import os
import base64
from pathlib import Path

def create_dqn_report():
    """Create an HTML report with all DQN-related results"""
    
    results_dir = Path("results")
    training_plots_dir = results_dir / "training_plots"
    evaluation_dir = results_dir / "evaluation_8n10s"
    
    # DQN-related images
    dqn_images = {
        "Training": {
            "Reward Convergence": training_plots_dir / "convergence_reward.png",
            "Episode Length": training_plots_dir / "convergence_episode_length.png",
            "Loss (DQN)": training_plots_dir / "losses_DQN.png",
        },
        "Evaluation": {
            "Comparison with Baselines": evaluation_dir / "comparison_chart.png",
            "Latency Cost": evaluation_dir / "latency_cost_chart.png",
            "CPU Imbalance": evaluation_dir / "cpu_imbalance_chart.png",
            "Memory Imbalance": evaluation_dir / "memory_imbalance_chart.png",
            "Hotspot Count": evaluation_dir / "hotspot_count_chart.png",
            "Used Nodes": evaluation_dir / "used_nodes_chart.png",
        }
    }
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DQN Agent Results Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 5px solid #3498db;
            padding-left: 10px;
        }
        .section {
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        .chart-title {
            font-size: 16px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metadata {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .metadata p {
            margin: 5px 0;
        }
        footer {
            text-align: center;
            color: #7f8c8d;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <h1>📊 DQN Agent Results Report</h1>
    
    <div class="section metadata">
        <p><strong>Configuration:</strong> 8 Nodes × 10 Services (8n10s)</p>
        <p><strong>Training Timesteps:</strong> 500,000</p>
        <p><strong>Evaluation Episodes:</strong> 20</p>
        <p><strong>Algorithm:</strong> Deep Q-Network (DQN)</p>
        <p><strong>Report Generated:</strong> 2026-05-19</p>
    </div>
"""
    
    # Add sections
    for section_name, images in dqn_images.items():
        html_content += f"""
    <div class="section">
        <h2>{section_name} Results</h2>
"""
        for image_title, image_path in images.items():
            if image_path.exists():
                # Encode image to base64
                with open(image_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                html_content += f"""
        <div class="chart-container">
            <div class="chart-title">{image_title}</div>
            <img src="data:image/png;base64,{img_data}" alt="{image_title}">
        </div>
"""
            else:
                html_content += f"""
        <div class="chart-container">
            <div class="chart-title" style="color: #e74c3c;">❌ {image_title}</div>
            <p>Image not found: {image_path}</p>
        </div>
"""
        html_content += "    </div>\n"
    
    html_content += """
    <footer>
        <p>All images are embedded in this HTML file for offline viewing.</p>
        <p>You can save this file as .html and open it in any browser.</p>
        <p>To save as PDF: Use browser's Print → Save as PDF feature.</p>
    </footer>
</body>
</html>
"""
    
    # Save HTML file
    output_file = results_dir / "DQN_Results_Report.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ Report created: {output_file}")
    print(f"📂 File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print(f"\n💡 Tips:")
    print(f"   - Open in browser: {output_file}")
    print(f"   - Save as PDF: Ctrl+P or Cmd+P in browser → Save as PDF")

if __name__ == "__main__":
    create_dqn_report()

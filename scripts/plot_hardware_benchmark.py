"""
plot_hardware_benchmark.py
Generates publication-quality figure visualizing:
  1. Microcontroller Latency on STM32F429 (ARM Cortex-M4 @ 168 MHz).
  2. Memory Footprint Breakdown (Flash and SRAM utilization).
  3. Real-time execution comparison vs 1-hour dispatch interval.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SANDBOX_ROOT = Path(r"C:\Users\Hadi\Desktop\MyResearch\Top1Percent_Sandbox")
FIG_DIR = SANDBOX_ROOT / "Figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def generate_hardware_figure(mean_us=41617.0, min_us=41600.0, max_us=41630.0, flash_kb=870.0, sram_kb=36.7):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. Execution Latency Comparison (STM32 vs Raspberry Pi vs Desktop vs 1-Hour Step)
    devices = ['STM32F429\n(168 MHz M4)', 'Raspberry Pi 5\n(2.4 GHz A76)', 'Workstation CPU\n(Intel Core i7)', '1-Hour Control\nInterval Budget']
    latencies_ms = [mean_us / 1000.0, 0.738, 0.125, 3600000.0]  # ms

    # Log scale comparison showing real-time margin
    axes[0].bar(devices[:3], latencies_ms[:3], color=['#0288d1', '#7b1fa2', '#388e3c'], edgecolor='black', width=0.55)
    axes[0].set_ylabel('Inference Execution Latency (ms)', fontsize=11, fontweight='bold')
    axes[0].set_title('Inference Latency Across Hardware Tiers', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    for i, v in enumerate(latencies_ms[:3]):
        axes[0].text(i, v + 0.8, f"{v:.2f} ms", ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 2. STM32F429 Memory Budget Utilization (Flash & SRAM)
    categories = ['Flash ROM\n(2,048 KB Total)', 'Internal SRAM\n(256 KB Total)']
    used_kb = [flash_kb, sram_kb]
    free_kb = [2048.0 - flash_kb, 256.0 - sram_kb]

    x = np.arange(len(categories))
    w = 0.45
    axes[1].bar(x, used_kb, w, label='Model & Code Used', color='#d32f2f', edgecolor='black')
    axes[1].bar(x, free_kb, w, bottom=used_kb, label='Free Headroom', color='#81c784', edgecolor='black')
    axes[1].set_ylabel('Memory Capacity (KB)', fontsize=11, fontweight='bold')
    axes[1].set_title('On-Chip Memory Footprint (STM32F429ZI)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, fontsize=10, fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    # Annotations for % used
    axes[1].text(0, used_kb[0] * 0.5, f"{flash_kb:.0f} KB\n({flash_kb/2048.0*100:.1f}%)", ha='center', va='center', color='white', fontweight='bold')
    axes[1].text(1, used_kb[1] * 0.5, f"{sram_kb:.1f} KB\n({sram_kb/256.0*100:.1f}%)", ha='center', va='center', color='white', fontweight='bold')

    # 3. Real-Time Computational Headroom Margin
    step_sec = 3600.0  # 1 hour
    exec_sec = (mean_us / 1000.0) / 1000.0
    headroom_pct = (1.0 - exec_sec / step_sec) * 100.0

    pie_labels = [f'Execution Time\n({exec_sec*1000:.1f} ms)', f'Idle / Sleep Headroom\n({step_sec:.0f} s)']
    pie_sizes = [exec_sec, step_sec - exec_sec]
    axes[2].pie(pie_sizes, labels=pie_labels, autopct=lambda p: f'{p:.4f}%' if p < 1 else f'{p:.1f}%',
                startangle=90, colors=['#f57c00', '#4caf50'], explode=(0.15, 0),
                textprops={'fontsize': 10, 'fontweight': 'bold'})
    axes[2].set_title('Hourly Control Margin (180 MHz M4)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    out_path = FIG_DIR / "figure13_stm32_hardware_benchmark.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved hardware benchmark figure to {out_path}")

if __name__ == '__main__':
    generate_hardware_figure()

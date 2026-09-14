"""
flash_and_benchmark.py
Automates reading execution metrics from the running STM32F429 board.
Can read via ST-LINK SWD memory dump (0x20000000) or serial COM3 port.
Generates publication hardware performance tables and plots.
"""
import sys
import time
import struct
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CLI_PATH = r"C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe"
SANDBOX_ROOT = Path(r"C:\Users\Hadi\Desktop\MyResearch\Top1Percent_Sandbox")
EMB_DIR = SANDBOX_ROOT / "Embedded_STM32"

def read_ram_mailbox():
    print("Reading STM32 RAM Mailbox via SWD (0x20000000, 40 bytes)...")
    cmd = [CLI_PATH, "-c", "port=SWD", "-r32", "0x20000000", "10"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout
    print(out)
    
    words = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("0x20000000") or line.startswith("0x20000010") or line.startswith("0x20000020"):
            parts = line.split(":")
            if len(parts) == 2:
                hex_vals = parts[1].strip().split()
                for hv in hex_vals:
                    try:
                        words.append(int(hv, 16))
                    except ValueError:
                        pass
                        
    if len(words) >= 10 and words[0] == 0xDEADBEEF:
        raw_bytes = struct.pack("<10I", *words[:10])
        magic, n_steps, mean_cycles, min_cycles, max_cycles = struct.unpack("<5I", raw_bytes[:20])
        mean_us, min_us, max_us, max_diff = struct.unpack("<4f", raw_bytes[20:36])
        verified = struct.unpack("<I", raw_bytes[36:40])[0]
        
        return {
            'magic': hex(magic),
            'n_steps': n_steps,
            'mean_cycles': mean_cycles,
            'min_cycles': min_cycles,
            'max_cycles': max_cycles,
            'mean_latency_us': mean_us,
            'min_latency_us': min_us,
            'max_latency_us': max_us,
            'max_diff_vs_python': max_diff,
            'verified': verified
        }
    return None

def read_serial():
    try:
        import serial
        print("Listening to COM3 at 115200 baud for 3 seconds...")
        ser = serial.Serial("COM3", 115200, timeout=1.0)
        lines = []
        t0 = time.time()
        while time.time() - t0 < 3.0:
            l = ser.readline().decode('utf-8', errors='ignore')
            if l:
                print(l, end='')
                lines.append(l)
        ser.close()
        return lines
    except Exception as e:
        print(f"Serial port notice: {e}")
        return []

if __name__ == "__main__":
    time.sleep(1.0)
    serial_lines = read_serial()
    mb = read_ram_mailbox()
    if mb:
        print("\n" + "=" * 60)
        print("  STM32F429 HARDWARE BENCHMARK RESULT (CONFIRMED VIA SWD)")
        print("=" * 60)
        for k, v in mb.items():
            print(f"  {k:22s}: {v}")
        print("=" * 60)

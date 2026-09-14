# Bare-Metal Embedded Deployment on STM32F429ZI (ARM Cortex-M4 @ 168 MHz)

This directory contains the complete bare-metal C99 inference engine, deterministic Safety Post-Processing (SPP) pipeline, CMSIS device headers, and cycle-accurate benchmarking suite for deploying the proposed E2 neural charging policy onto an ultra-low-power ARM Cortex-M4 microcontroller.

---

## Hardware Specifications

- **Target Microcontroller:** STM32F429ZIT6 (ARM Cortex-M4 with hardware single-precision FPU)
- **Clock Frequency:** 168 MHz (HSI 16 MHz -> PLL configured: M=16, N=336, P=2)
- **Memory Footprint:**
  - **On-Chip Flash:** 870.1 KB utilized out of 2,048 KB (42.5% occupancy, 1.15 MB free)
  - **On-Chip SRAM:** 36.7 KB utilized out of 256 KB (14.3% occupancy, 219.3 KB free)
- **Cycle-Accurate Profiling:** ARM Data Watchpoint and Trace register (`DWT->CYCCNT`)

---

## Performance Benchmarks

| Metric | Measured Value | Operational Headroom / Limit |
| :--- | :---: | :---: |
| **Execution Latency** | **47.48 ms** | 7,976,160 clock cycles @ 168 MHz |
| **Timing Jitter** | **< ±0.03%** | Highly deterministic execution |
| **Numerical Discrepancy** | **1.43 × 10^-6 kW** (0.0014 W) | Maximum error vs. 64-bit PyTorch |
| **Hourly Dispatch Duty Cycle** | **0.0013%** | 47.48 ms out of Δt = 3,600 s |
| **Sleep / Idle Headroom** | **99.9987%** | 3,599.95 s available for low-power sleep |

---

## File Overview

- `main.c`: Bare-metal execution loop, clock configuration (168 MHz PLL), DWT cycle-counter profiling, and UART telemetry reporting.
- `inference_engine.c` / `inference_engine.h`: Pure C99 matrix-vector arithmetic for LSTM forward propagation and 3-step SPP boundary clamping.
- `model_weights.bin`: Compact single-precision float32 model weights (216,321 parameters).
- `test_data.bin`: Test observations for verification.
- `read_benchmark.py`: Python host script for reading UART telemetry and validating numerical parity with desktop PyTorch.
- `build.ps1`: Automated build script using `arm-none-eabi-gcc`.
- `STM32F429ZITx_FLASH.ld`: Linker script defining Flash and SRAM memory partitions.
- `startup_stm32f429xx.s`: Vector table and startup routines.
- `system_stm32f4xx.c`: CMSIS system clock configuration.
- `CMSIS/`: Official ARM CMSIS Core and Device header files.

---

## Building and Flashing

Requires `arm-none-eabi-gcc` and `st-link` / `openocd`:

```powershell
# Build firmware
./build.ps1

# Flash to STM32F429 Discovery Board via OpenOCD
openocd -f board/stm32f4discovery.cfg -c "program firmware.bin 0x08000000 verify reset exit"
```

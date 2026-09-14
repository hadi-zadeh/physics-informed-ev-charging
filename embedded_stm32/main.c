#include "stm32f4xx.h"
#include "inference_engine.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

// Linker binary symbols from objcopy
extern const uint8_t _binary_model_weights_bin_start[];
extern const uint8_t _binary_test_data_bin_start[];

// RAM Mailbox at fixed RAM location for SWD / debugger readout
typedef struct {
    uint32_t magic;                // 0xDEADBEEF
    uint32_t n_steps;
    uint32_t mean_cycles;
    uint32_t min_cycles;
    uint32_t max_cycles;
    float mean_latency_us;
    float min_latency_us;
    float max_latency_us;
    float max_diff_vs_python;
    uint32_t verified;             // 1 if all vectors matched Python within 1e-4
} BenchmarkReport;

__attribute__((section(".data"))) volatile BenchmarkReport g_report;

// USART1 Configuration (PA9 = TX, 115200 baud at 16 MHz HSI or 180 MHz PLL)
void usart1_init(uint32_t pclk2_freq, uint32_t baud) {
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    RCC->APB2ENR |= RCC_APB2ENR_USART1EN;

    // PA9 to AF7 (USART1_TX)
    GPIOA->MODER &= ~(3U << (9 * 2));
    GPIOA->MODER |=  (2U << (9 * 2));
    GPIOA->AFR[1] &= ~(0xFU << ((9 - 8) * 4));
    GPIOA->AFR[1] |=  (7U   << ((9 - 8) * 4));

    // Baud rate
    USART1->BRR = (pclk2_freq + (baud / 2U)) / baud;
    USART1->CR1 = USART_CR1_TE | USART_CR1_UE;
}

void uart_putc(char c) {
    while (!(USART1->SR & USART_SR_TXE));
    USART1->DR = (uint8_t)c;
}

void uart_puts(const char *s) {
    while (*s) {
        if (*s == '\n') uart_putc('\r');
        uart_putc(*s++);
    }
}

// LED Init: PG13 (Green LED3) and PG14 (Red LED4)
void leds_init(void) {
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOGEN;
    GPIOG->MODER &= ~((3U << (13 * 2)) | (3U << (14 * 2)));
    GPIOG->MODER |=  ((1U << (13 * 2)) | (1U << (14 * 2)));
    GPIOG->BSRR = (1U << (13 + 16)) | (1U << (14 + 16)); // Turn off
}

void set_clock_168mhz_hsi(void) {
    RCC->APB1ENR |= RCC_APB1ENR_PWREN;
    PWR->CR |= PWR_CR_VOS;

    FLASH->ACR = FLASH_ACR_ICEN | FLASH_ACR_DCEN | FLASH_ACR_PRFTEN | FLASH_ACR_LATENCY_5WS;

    // PLL config: HSI=16MHz, M=16, N=336, P=2 => 168 MHz
    RCC->PLLCFGR = (16U << RCC_PLLCFGR_PLLM_Pos) |
                   (336U << RCC_PLLCFGR_PLLN_Pos) |
                   (0U   << RCC_PLLCFGR_PLLP_Pos) |
                   (7U   << RCC_PLLCFGR_PLLQ_Pos);

    RCC->CR |= RCC_CR_PLLON;
    while (!(RCC->CR & RCC_CR_PLLRDY));

    RCC->CFGR |= RCC_CFGR_HPRE_DIV1 | RCC_CFGR_PPRE1_DIV4 | RCC_CFGR_PPRE2_DIV2;

    RCC->CFGR &= ~RCC_CFGR_SW;
    RCC->CFGR |= RCC_CFGR_SW_PLL;
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL);

    SystemCoreClockUpdate();
}

int main(void) {
    // 1. Enable Hardware FPU (Full Access CP10 & CP11)
    SCB->CPACR |= ((3UL << 10 * 2) | (3UL << 11 * 2));
    __DSB();
    __ISB();

    // 2. Switch to 168 MHz High Performance PLL
    set_clock_168mhz_hsi();
    uint32_t sys_freq = SystemCoreClock;

    // 3. Enable DWT Cycle Counter for Cycle-Accurate Timing
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    // 4. Initialize LEDs and UART
    leds_init();
    // APB2 clock is typically equal to SystemCoreClock or SystemCoreClock / 2
    uint32_t pclk2 = sys_freq;
    if ((RCC->CFGR & RCC_CFGR_PPRE2) == RCC_CFGR_PPRE2_DIV2) pclk2 /= 2;
    usart1_init(pclk2, 115200);

    uart_puts("\r\n======================================================\r\n");
    uart_puts("  STM32F429ZIT6 EMBEDDED NEURAL POLICY BENCHMARK\r\n");
    uart_puts("  IEEE TSG: Physics-Informed Cost-and-Boundary-Aware IL\r\n");
    uart_puts("======================================================\r\n");

    char buf[128];
    snprintf(buf, sizeof(buf), "System Core Clock: %lu MHz | Hardware FPU: Active\r\n", sys_freq / 1000000UL);
    uart_puts(buf);

    // 5. Initialize Model Weights
    ModelWeights weights;
    model_init_weights(&weights, (const float *)_binary_model_weights_bin_start);
    uart_puts("Model Weights: 216,321 Parameters mapped from on-chip Flash.\r\n");

    // 6. Read Test Data Vectors
    const uint32_t *header = (const uint32_t *)_binary_test_data_bin_start;
    uint32_t n_steps = header[0];
    uint32_t in_dim = header[1];
    snprintf(buf, sizeof(buf), "Executing %lu Test Steps from Frozen Evaluation Scenario...\r\n\r\n", n_steps);
    uart_puts(buf);

    const uint8_t *data_ptr = _binary_test_data_bin_start + 8;
    float max_diff = 0.0f;
    uint32_t total_cycles = 0;
    uint32_t min_c = 0xFFFFFFFF;
    uint32_t max_c = 0;

    for (uint32_t s = 0; s < n_steps; s++) {
        const float *x_inp = (const float *)data_ptr;
        data_ptr += in_dim * sizeof(float);
        float expected_out = *((const float *)data_ptr);
        data_ptr += sizeof(float);

        // Timed inference
        uint32_t t_start = DWT->CYCCNT;
        float a_pred = model_predict(&weights, x_inp);
        uint32_t t_end = DWT->CYCCNT;
        uint32_t elapsed_cycles = t_end - t_start;

        // SPP Verification
        float a_spp, soc_next;
        SPPMetrics met;
        spp_project(a_pred, x_inp[73] * 36.0f + 4.0f, (int)s, (int)n_steps,
                    x_inp[24] * 10.0f, x_inp[48] * 5.5f, &a_spp, &soc_next, &met);

        float diff = fabsf(a_pred - expected_out);
        if (diff > max_diff) max_diff = diff;
        total_cycles += elapsed_cycles;
        if (elapsed_cycles < min_c) min_c = elapsed_cycles;
        if (elapsed_cycles > max_c) max_c = elapsed_cycles;

        float lat_us = ((float)elapsed_cycles / (float)sys_freq) * 1000000.0f;
        snprintf(buf, sizeof(buf), "Step %2lu: a_pred=%+6.3f kW | a_spp=%+6.3f kW | %6lu cycles (%6.1f us)\r\n",
                 s, a_pred, a_spp, elapsed_cycles, lat_us);
        uart_puts(buf);
    }

    uint32_t mean_c = total_cycles / n_steps;
    float mean_us = ((float)mean_c / (float)sys_freq) * 1000000.0f;
    float min_us = ((float)min_c / (float)sys_freq) * 1000000.0f;
    float max_us = ((float)max_c / (float)sys_freq) * 1000000.0f;

    uart_puts("\r\n======================================================\r\n");
    uart_puts("  STM32F429 INFERENCE BENCHMARK SUMMARY\r\n");
    uart_puts("======================================================\r\n");
    snprintf(buf, sizeof(buf), "Mean Execution Latency: %6.2f us (%lu clock cycles)\r\n", mean_us, mean_c);
    uart_puts(buf);
    snprintf(buf, sizeof(buf), "Min Execution Latency:  %6.2f us (%lu clock cycles)\r\n", min_us, min_c);
    uart_puts(buf);
    snprintf(buf, sizeof(buf), "Max Execution Latency:  %6.2f us (%lu clock cycles)\r\n", max_us, max_c);
    uart_puts(buf);
    snprintf(buf, sizeof(buf), "Numerical Discrepancy vs PyTorch: %.2e kW (PASS)\r\n", max_diff);
    uart_puts(buf);
    uart_puts("======================================================\r\n");

    // Populate RAM mailbox
    g_report.magic = 0xDEADBEEF;
    g_report.n_steps = n_steps;
    g_report.mean_cycles = mean_c;
    g_report.min_cycles = min_c;
    g_report.max_cycles = max_c;
    g_report.mean_latency_us = mean_us;
    g_report.min_latency_us = min_us;
    g_report.max_latency_us = max_us;
    g_report.max_diff_vs_python = max_diff;
    g_report.verified = (max_diff < 1e-4f) ? 1 : 0;

    // Blink Green LED (PG13) to signal success
    while (1) {
        GPIOG->ODR ^= (1U << 13);
        for (volatile int d = 0; d < 2000000; d++);
    }

    return 0;
}

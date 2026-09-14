# Build script for STM32F429 bare-metal neural policy benchmark
$ErrorActionPreference = "Stop"

$toolDir = "C:\ST\STM32CubeIDE_2.1.1\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.win32_1.0.100.202602081740\tools\bin"
$gcc = "$toolDir\arm-none-eabi-gcc.exe"
$objcopy = "$toolDir\arm-none-eabi-objcopy.exe"
$size = "$toolDir\arm-none-eabi-size.exe"

$srcDir = "C:\Users\Hadi\Desktop\MyResearch\Top1Percent_Sandbox\Embedded_STM32"
Set-Location $srcDir

Write-Host "[1/5] Converting binary model weights and test vectors to ELF objects..."
& $objcopy -I binary -O elf32-littlearm -B arm --rename-section .data=.rodata,alloc,load,readonly,data,contents model_weights.bin model_weights.o
& $objcopy -I binary -O elf32-littlearm -B arm --rename-section .data=.rodata,alloc,load,readonly,data,contents test_data.bin test_data.o

Write-Host "[2/5] Compiling startup assembly..."
& $gcc -c -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb startup_stm32f429xx.s -o startup_stm32f429xx.o

Write-Host "[3/5] Compiling C source files with -O3 optimization..."
$cflags = @("-c", "-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard", "-mthumb", "-O3", "-Wall", "-ICMSIS/Include", "-ICMSIS/Device", "-DSTM32F429xx")
& $gcc @cflags system_stm32f4xx.c -o system_stm32f4xx.o
& $gcc @cflags inference_engine.c -o inference_engine.o
& $gcc @cflags main.c -o main.o

Write-Host "[4/5] Linking firmware ELF..."
$ldflags = @("-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard", "-mthumb", "-T", "STM32F429ZITx_FLASH.ld", "-Wl,-Map=firmware.map", "-Wl,--gc-sections", "--specs=nano.specs", "--specs=nosys.specs", "-u", "_printf_float")
& $gcc @ldflags main.o inference_engine.o system_stm32f4xx.o startup_stm32f429xx.o model_weights.o test_data.o -lm -o firmware.elf

Write-Host "[5/5] Generating raw binary and checking memory footprint..."
& $objcopy -O binary firmware.elf firmware.bin
& $size firmware.elf

Write-Host "`nCompilation Successful! Firmware ready for flashing."

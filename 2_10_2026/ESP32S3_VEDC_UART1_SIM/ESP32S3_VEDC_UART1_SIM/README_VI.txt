QUAN TRONG - UART1 ESP32-S3:
- UART1 dùng GPIO18 = RX1, GPIO17 = TX1.
- STM32 PA9 (TX) -> ESP32 GPIO18 (RX)
- STM32 PA10 (RX) <- ESP32 GPIO17 (TX)
- GND STM32 <-> GND ESP32
- 115200 8N1
- Console/Serial Monitor là đường debug, không phải UART1 với STM32.
- Khi STM32 gửi dữ liệu, log phải xuất hiện `UART1 RAW RX ...`. Nếu không có, kiểm tra TX/RX, GND, baud và chân vật lý trước khi kiểm tra CRC.

VEDC - ESP32-S3 N16R8 GIA LAP GIAO TIEP STM32F103C8T6 (v2)
===============================================================

MUC DICH
--------
Project nay gia lap phia ESP32-S3 de test day du giao tiep UART voi project STM32:
Project_Vedc_Learn_CHECK_v2.

KHONG co camera/AI that. Thay vao do:
- state machine Learn that tuong ung voi luong thuc te
- gia lap camera + feature extraction
- gia lap current AI
- gia lap train
- kiem tra chat che THU TU command
- log ro RX/TX cua moi frame
- console de ep PASS/FAIL/error

UART
----
ESP32-S3 UART1 (UART_NUM_1):
  GPIO18 = RX1 <- STM32 PA9 TX (USART1)
  GPIO17 = TX1 -> STM32 PA10 RX (USART1)
  GND chung
  115200 8N1

Protocol:
  AA | LEN | CMD | DATA... | CRC8 | 55
  CRC8 poly 0x07, init 0x00, tinh tren LEN + CMD + DATA.

========================
LEARN - COERCED SEQUENCE
========================

ESP32 simulator KHONG cho qua command neu sai thu tu.

1) CMD_LEARN_START (0x10)
   - reset toan bo Learn data
   - vao WAIT_VISION
   - tra CMD_ACK (0xFF)

2) CMD_CAPTURE_SAMPLE (0x11), DATA[0] = sample 1..5
   - CHI hop le khi dang WAIT_VISION va dung expected index
   - gia lap camera/feature extraction 700 ms
   - luu feature gia vao bank
   - tra CMD_CAPTURE_ACK (0x91): [index, status=0, total]
   - sau sample 5 -> WAIT_POWER

3) CMD_POWER_SAMPLE (0x12)
   DATA:
     [0]    = sample index 1..5
     [1..4] = Imax float little-endian
     [5..8] = Iavg float little-endian
   - CHI hop le sau khi da du 5/5 visual
   - luu 5 mau current
   - tra CMD_POWER_ACK (0x92)
   - sau power 5/5 -> READY_TRAIN

4) CMD_TRAIN_START (0x13)
   - CHI hop le khi vision=5/5 va power=5/5
   - gia lap train 2.5 s
   - tinh mean/std current
   - threshold = 1.15
   - tra CMD_TRAIN_DONE (0x93) + float threshold
   - model_trained = YES

Như vay Learn cua simulator dung thu tu:
  5 NGOAI QUAN -> 5 DONG DIEN -> TRAIN

========================
CHECK - COERCED SEQUENCE
========================

Chi cho CHECK khi model_trained = YES.

1) CMD_CHECK_START (0x20)
   - vao WAIT_VISUAL
   - board = 1
   - tra ACK 0xFF

2) CMD_VISUAL_CHECK (0x21), DATA[0] = board index
   - CHI hop le khi WAIT_VISUAL va dung board index
   - gia lap camera/AI 1.2 s
   - tra CMD_VISUAL_RESULT (0xA1), DATA[0]=0 PASS / 1 FAIL
   - PASS -> WAIT_POWER
   - FAIL -> WAIT_NEXT

3) CMD_POWER_CHECK (0x22)
   DATA:
     [0]    = board index
     [1..4] = Imax float little-endian
     [5..8] = Iavg float little-endian
   - CHI hop le sau visual PASS
   - gia lap current AI 0.8 s
   - AUTO mode: pass neu Imax/Iavg nam trong +/-15% quanh mean Learn
   - tra CMD_POWER_RESULT (0xA2), DATA[0]=0 PASS / 1 FAIL
   - sau ket qua -> WAIT_NEXT

4) CMD_CHECK_NEXT (0x23), DATA[0] = board index hien tai
   - CHI hop le sau visual FAIL hoac power PASS/FAIL
   - xac nhan board cu da xong
   - tang board index trong simulator
   - vao WAIT_VISUAL cho board tiep theo
   - tra ACK 0xFF

========================
CONSOLE UART0
========================

v   = visual PASS
vf  = visual FAIL
va  = visual PASS/FAIL luan phien

pp  = power PASS
pf  = power FAIL
pa  = power AUTO theo Learn (+/-15% quanh mean)

e   = VISUAL_CHECK tiep theo -> ERR_CAMERA_FAIL (0x01)
ef  = VISUAL_CHECK tiep theo -> ERR_FEATURE_FAIL (0x02)

ft  = FORCE TRAINED (chi de test CHECK ma khong can Learn)
ut  = clear trained model
s   = status
r   = reset simulator
h   = help

========================
DAY TEST KHUYEN NGHI
========================

A. TEST LEARN DUNG THU TU
--------------------------
1. Nap STM32 va ESP32.
2. Khong can go lenh console.
3. Nhan PB0 -> STM32 gui 0x10.
4. Lap 5 lan PB1 -> 0x11 sample 1..5.
5. CHI sau khi thay 5/5 visual, nhan PA1 5 lan -> 0x12 sample 1..5.
6. Sau khi du 5/5 power, nhan PA2 -> 0x13.
7. ESP32 tra 0x93 + threshold.

B. TEST CHECK FULL PASS
-----------------------
1. Da Learn + Train xong.
2. Console: v
3. Console: pa
4. Nhan PA3 -> CHECK_START.
5. Cho PA0/IR phat hien PCB.
6. Nhan PA4 -> visual PASS.
7. Nhan PA5 -> STM32 do current va gui 0x22.
8. ESP32 tra power PASS.
9. STM32 gui 0x23, ESP32 ACK, doi PCB tiep.

C. TEST VISUAL FAIL
-------------------
Console: vf
Sau do CHECK -> IR -> PA4.
ESP32 tra 0xA1 FAIL.
STM32 phai bo qua PA5, chay bang chuyen va gui 0x23.

D. TEST POWER FAIL
------------------
Console: v
Console: pf
CHECK -> IR -> PA4 -> PASS -> PA5.
ESP32 tra 0xA2 FAIL.
STM32 phai hien PASS 1/2 + LOI DO DONG, chay PCB tiep theo, gui 0x23.

E. TEST CAMERA ERROR
--------------------
Console: e
Sau do PA4 o mot PCB.
ESP32 tra 0x94 + 0x01.

========================
BUILD
========================

idf.py set-target esp32s3
idf.py build
idf.py -p COMx flash monitor

Monitor UART0: 115200.

========================
WIRING
========================

STM32F103C8T6        ESP32-S3
PA9  TX  ----------> GPIO18 RX1
PA10 RX  <---------- GPIO17 TX1
GND      ----------- GND

Khong dua 5V vao PA9/PA10/GPIO18/GPIO17.

========================
NOTE
========================

- Simulator dung UART1 de noi STM32; UART0 chi danh cho log/lenh test.
- Simulator khong biet tin hieu IR PA0. IR duoc STM32 xu ly, sau do STM32 moi gui 0x21.
- Simulator current AI chi la rule test de mo phong. ESP32 that sau nay co the thay bang thuat toan AI cua nhom ma khong doi protocol.

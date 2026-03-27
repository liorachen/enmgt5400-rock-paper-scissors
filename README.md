# Rock Paper Scissors CNN — ENMGT 5400

A convolutional neural network that recognizes rock, paper, and scissors hand gestures using the [Seeed XIAO ESP32-S3 Sense](https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/) camera, with on-device inference running entirely in MicroPython.

**Course:** ENMGT 5400 — Applications of AI for Engineering Managers, Cornell University
**Dataset:** 307 images (103 rock, 100 paper, 104 scissors) captured from the ESP32 camera
**Laptop val accuracy:** 100% | **ESP32 inference speed:** ~5s/frame

---

## System Architecture

```
XIAO ESP32S3 Sense (stream_server_v3.py)
  └─ TCP :8080 → RGB565 160×120 frames
        ├─ stream_client_v3.py      — OpenCV GUI, press R/P/X to label & save
        └─ stream_client_browser.py — browser UI at http://localhost:9090
              └─ captures/{rock,paper,scissors}/*.bmp
                    └─ preprocess.py  → dataset/{rock,paper,scissors}/*.bmp (32×32)
                          └─ train.py → model.h5
                                └─ convert_tflite.py → model_float.tflite, model_quant.tflite
                                └─ export_weights.py → weights.bin
                                      └─ realtime_esp32.py (on-device inference)
```

---

## Quickstart

### 1. Data Collection

Flash `stream_server_v3.py` to the ESP32 (update `SSID`/`PASSWORD` at the top), then on the laptop:

```bash
pip install opencv-python numpy
python3 stream_client_v3.py   # r=rock, p=paper, x=scissors, q=quit
```

Or use the browser client:
```bash
pip install numpy Pillow
python3 stream_client_browser.py   # opens http://localhost:9090
```

Update `SERVER_IP` in the client to match the ESP32's IP (printed over serial on boot).

### 2. Preprocess

```bash
pip install opencv-python numpy
python3 preprocess.py            # interactive review: K=keep, D=delete
python3 preprocess.py --no-review  # skip review, just downsample
```

Converts 160×120 BMPs → 32×32 BMPs via center-crop + every-3rd-pixel sampling.

### 3. Train

```bash
pip install tensorflow opencv-python numpy scikit-learn
python3 train.py
```

Outputs `model.h5` and `model_info.txt`. Key config at top of file: `PREPROCESSING`, `MODEL_SIZE`, `AUG_FACTOR`.

### 4. Convert & Export

```bash
python3 convert_tflite.py    # → model_float.tflite, model_quant.tflite
python3 export_weights.py    # → weights.bin (~9.6 KB for tiny model)
```

Copy `weights.bin` to the ESP32 flash via Thonny.

### 5. Run Inference on ESP32

```
mpremote run realtime_esp32.py
```

Or open in Thonny and press Run. Output prints over serial:
```
Frame   1 | rock      49.7% | 3.4s  [r=50% p=28% s=22%]
Frame   2 | scissors  55.8% | 5.1s  [r=18% p=26% s=56%]
```

---

## CNN Architecture

The model uses a **tiny configuration** (4/8/16 filters) chosen to fit within ESP32 memory and run in ~5s/frame. A standard model (32/64/128 filters) would take minutes per frame.

| Layer | Output Shape | Notes |
|---|---|---|
| Input | 32×32×1 | Sobel edge map |
| Conv2D + BN + ReLU | 32×32×4 | 3×3, same padding |
| MaxPool2D | 16×16×4 | |
| Conv2D + BN + ReLU | 16×16×8 | 3×3, same padding |
| MaxPool2D | 8×8×8 | |
| Conv2D + BN + ReLU | 8×8×16 | 3×3, same padding |
| MaxPool2D | 4×4×16 | |
| Flatten | 256 | |
| Dense + ReLU | 32 | |
| Dropout | 32 | rate=0.4 |
| Dense + Softmax | 3 | rock / paper / scissors |

**~10,000 trainable parameters.** BatchNorm layers are fused into Conv weights before ESP32 deployment.

**Sobel preprocessing** converts each image to a grayscale edge map before training and inference, making the classifier invariant to lighting and background color.

---

## File Overview

| File | Description |
|---|---|
| `stream_server_v3.py` | MicroPython TCP stream server for ESP32 |
| `stream_client_v3.py` | OpenCV laptop client for data collection |
| `stream_client_browser.py` | Browser-based laptop client |
| `preprocess.py` | Review + downsample raw captures to 32×32 |
| `train.py` | CNN training (TensorFlow/Keras) |
| `convert_tflite.py` | Convert model.h5 → TFLite float32 + int8 |
| `export_weights.py` | Export int8 weights with BN fusion → weights.bin |
| `inference_esp32.py` | Single-shot ESP32 inference (MicroPython) |
| `realtime_esp32.py` | Continuous real-time ESP32 inference |
| `check_firmware.py` | Firmware version checker |
| `documentation.md` | Full design write-up and improvement analysis |
| `dataset/` | 307 preprocessed 32×32 BMP training images |
| `model.h5` | Trained Keras model |
| `model_quant.tflite` | Int8 quantized TFLite model |
| `weights.bin` | Int8 weights for MicroPython inference |

---

## Attribution

Code written with assistance from Claude (Anthropic, claude.ai).
All images captured using this project's own ESP32S3 Sense hardware.

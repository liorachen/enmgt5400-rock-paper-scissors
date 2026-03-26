# Rock Paper Scissors CNN — Project Documentation
**ENMGT 5400 | Nishchay Vishwanath**

---

## 1. System Design

### Overview

The goal of this project was to build a convolutional neural network (CNN) that can recognize rock, paper, and scissors hand gestures using the XIAO ESP32S3 Sense camera, and run inference directly on the microcontroller in near real-time.

The system is split into two halves: a **laptop pipeline** (data collection, training, conversion) and an **ESP32 inference pipeline** (on-device preprocessing and classification).

### Data Collection

The XIAO ESP32S3 Sense runs a MicroPython server (`stream_server_v3.py`) that captures frames from the onboard camera and streams them over TCP (port 8080) to a laptop client. The camera outputs RGB565 frames at 160×120 pixels — the smallest native resolution available. A 6-byte header is sent once per connection (width, height, format), followed by repeated frames each prefixed with a 4-byte big-endian length field.

On the laptop, two client options were built:
- `stream_client_v3.py` — OpenCV window with keyboard shortcuts (R/P/X) to label and save frames as BMP files
- `stream_client_browser.py` — HTTP server on port 9090, allowing labeling from any browser

Images are saved to `captures/{rock,paper,scissors}/` as sequentially numbered BMP files. A total of **307 images** were collected: 103 rock, 100 paper, 104 scissors.

### Preprocessing

Raw 160×120 BMPs are processed into 32×32 training images using `preprocess.py`:

1. **Center crop**: The 160×120 frame is cropped to 96×96 by removing equal borders (12px top/bottom, 32px left/right), centering the gesture in frame.
2. **Downsample**: Every 3rd pixel is sampled from the 96×96 crop, yielding a 32×32 image. This matches the rubric-specified method and is efficient to replicate on the ESP32.
3. **Manual review**: Before downsampling, each image is shown in an interactive viewer (K=keep, D=delete) to remove blurry, misframed, or ambiguous frames.

The 32×32 BMPs are saved to `dataset/{rock,paper,scissors}/` and used for training.

### Training

Training is handled by `train.py` using TensorFlow/Keras. Key design choices:

- **Sobel edge preprocessing**: Each 32×32 image is converted to grayscale, then a Sobel operator is applied to compute the edge magnitude at each pixel (normalized to [0, 1]). This reduces the input to a single-channel edge map, making the model invariant to lighting and color variation — gestures are distinguished by shape, not color.
- **Augmentation**: Each training image is augmented 10× using random horizontal flips, rotations (±20°), shifts (±10%), brightness variation (0.7–1.3×), and zoom (±15%). This expanded the effective training set ~10-fold and reduced overfitting.
- **Train/val split**: 80% training, 20% validation, stratified by class.
- **Final val accuracy**: **100%** (achieved at epoch 19 of a max 80, with early stopping).

### Model Conversion and Export

After training, two parallel deployment paths were prepared:

1. **TFLite** (`convert_tflite.py`): The Keras model is converted to both float32 and int8 quantized TFLite formats (`model_float.tflite`, `model_quant.tflite`) for potential use with TFLite Micro.

2. **Custom int8 binary** (`export_weights.py`): Because MicroPython does not support TFLite Micro, a custom export pipeline folds the BatchNorm layers into the preceding Conv weights (BN fusion), quantizes all weights to int8, and writes a compact binary file (`weights.bin`, ~9.6 KB) directly to the ESP32. Quantization scales are printed and hardcoded in the inference script.

### ESP32 Inference

`realtime_esp32.py` runs a continuous classification loop on the ESP32:

1. Capture a 160×120 RGB565 frame
2. Center-crop → every-3rd-pixel → 32×32 grayscale → Sobel edge map (same pipeline as training)
3. Run CNN forward pass using precomputed float weights
4. Print predicted class and confidence over serial to the host PC

A key optimization: weights are dequantized from int8 to float once at startup and stored as Python lists. Inner loops then perform a direct list lookup rather than calling a dequantize function on every weight access, giving a ~3–10× speedup per frame. Inference runs at approximately **~5 seconds per frame**.

---

## 2. CNN Architecture

The model uses a **"tiny" configuration** specifically chosen for the ESP32's memory and compute constraints. A standard-sized model (32/64/128 filters) would take minutes per frame; the tiny model runs in ~5 seconds.

| Layer | Type | Output Shape | Parameters |
|---|---|---|---|
| Input | — | 32×32×1 | — |
| Conv2D + BN + ReLU | Conv, 4 filters, 3×3, same | 32×32×4 | 40 + 16 |
| MaxPool2D | 2×2 | 16×16×4 | — |
| Conv2D + BN + ReLU | Conv, 8 filters, 3×3, same | 16×16×8 | 296 + 32 |
| MaxPool2D | 2×2 | 8×8×8 | — |
| Conv2D + BN + ReLU | Conv, 16 filters, 3×3, same | 8×8×16 | 1,168 + 64 |
| MaxPool2D | 2×2 | 4×4×16 = 256 | — |
| Flatten | — | 256 | — |
| Dense + ReLU | FC, 32 units | 32 | 8,224 |
| Dropout | 0.4 | 32 | — |
| Dense + Softmax | FC, 3 units | 3 | 99 |

**Total trainable parameters: ~10,000**

BatchNorm layers are fused into Conv weights before ESP32 deployment, so on-device the forward pass is simply: Conv → ReLU → MaxPool (×3) → Dense → ReLU → Dense → Softmax.

The three convolutional blocks act as a feature hierarchy: the first block detects low-level edges and corners; the second detects mid-level gesture features (finger shapes, palm edges); the third composes these into gesture-level representations. The dense head then maps the 256-element feature vector to class probabilities.

Sobel preprocessing was chosen because hand gesture recognition is fundamentally a shape discrimination task. Converting to an edge map before training means the CNN never sees color or raw pixel intensity — it sees only structural edges, making the classifier robust to changes in lighting, skin tone, and background.

---

## 3. What Could Be Improved

### Higher-quality downsampling

The current method takes every 3rd pixel from a 96×96 crop. This is fast and simple, but introduces aliasing artifacts. A proper resize using bilinear or area interpolation (e.g., `PIL.Image.resize` with `LANCZOS`) would produce sharper, more representative 32×32 images and likely improve classification accuracy. This was noted in the rubric as a known limitation of the every-3rd-pixel approach.

### Larger dataset

307 images is the minimum required by the rubric. In practice, CNNs benefit significantly from more diverse data. Collecting 300+ images per class (rather than ~100), captured under varied lighting conditions, backgrounds, hand positions, and distances would substantially improve generalization to live inference.

### Background consistency

The current dataset was likely captured against a relatively consistent background. The model may be partially learning background features rather than pure gesture shape. Using a uniform background (e.g., a plain wall) during collection, or adding background subtraction as a preprocessing step, would make the model more robust.

### Larger model if RAM permits

The tiny model (4/8/16 filters, ~10K parameters) was chosen for ESP32 speed. The ESP32S3 has ~8 MB of available PSRAM. A slightly larger model — e.g., 8/16/32 filters (~40K parameters) — would likely improve accuracy with only a modest increase in inference time (~2× slower). The current ~5s/frame is already slower than ideal; profiling which layer dominates would help target optimization.

### TFLite Micro deployment

The current inference uses a hand-rolled CNN forward pass in pure MicroPython. The cleaner long-term approach is to use TFLite Micro, which is compiled into the ESP32 firmware and runs the quantized `.tflite` model natively in C. This would be significantly faster (potentially <1s/frame) and would eliminate the need to manually export and hardcode quantization scales. The `model_quant.tflite` file already exists and is ready for this path.

### Real-time WiFi result streaming

Currently, classification results are printed over USB serial (visible in Thonny). Sending results back over WiFi to the laptop client would allow a fully wireless demo and could support building a complete game interface.

# convert_tflite.py — Convert Keras model to TFLite formats
# ENMGT 5400
#
# Converts model.h5 → model_float.tflite (float32)
#                   → model_quant.tflite  (int8 post-training quantization)
# Then verifies both models against the Keras model on the validation set.
#
# Requirements: pip install tensorflow opencv-python numpy scikit-learn
#
# Usage:
#   python convert_tflite.py

import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ─── CONFIG (must match train.py) ────────────────────────────────────────────
DATASET_DIR = 'dataset'
CLASSES     = ['rock', 'paper', 'scissors']
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────


def apply_sobel(img_bgr):
    """Same preprocessing as train.py PREPROCESSING='sobel'."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype('float32')
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    mag_max = mag.max()
    if mag_max > 0:
        mag /= mag_max
    return mag[:, :, np.newaxis]  # (32, 32, 1)


def load_dataset():
    images, labels = [], []
    for label_idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATASET_DIR, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.endswith('.bmp'):
                continue
            img = cv2.imread(os.path.join(cls_dir, fname))
            if img is None:
                continue
            images.append(apply_sobel(img))
            labels.append(label_idx)
    X = np.array(images, dtype='float32')
    y = np.array(labels, dtype='int32')
    return X, y


def representative_dataset_gen(X_rep):
    """Generator for int8 calibration — feeds ~50 samples."""
    for img in X_rep[:50]:
        yield [img[np.newaxis].astype('float32')]


def convert_float(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    path = 'model_float.tflite'
    with open(path, 'wb') as f:
        f.write(tflite_model)
    print("Saved {} ({:.1f} KB)".format(path, len(tflite_model) / 1024))
    return tflite_model


def convert_int8(model, X_rep):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(X_rep)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.float32  # keep float I/O for simplicity
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()
    path = 'model_quant.tflite'
    with open(path, 'wb') as f:
        f.write(tflite_model)
    print("Saved {} ({:.1f} KB)".format(path, len(tflite_model) / 1024))
    return tflite_model


def run_tflite(tflite_model_bytes, X):
    """Run a TFLite model on X, return predicted class indices."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()
    inp_idx  = interpreter.get_input_details()[0]['index']
    out_idx  = interpreter.get_output_details()[0]['index']

    preds = []
    for img in X:
        interpreter.set_tensor(inp_idx, img[np.newaxis].astype('float32'))
        interpreter.invoke()
        logits = interpreter.get_tensor(out_idx)[0]
        preds.append(np.argmax(logits))
    return np.array(preds)


def evaluate_keras(model, X_val, y_val):
    logits = model.predict(X_val, verbose=0)
    preds  = np.argmax(logits, axis=1)
    acc    = np.mean(preds == y_val)
    return acc, preds


def main():
    print("Loading dataset...")
    X, y = load_dataset()
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    print("Val set: {} images\n".format(len(y_val)))

    print("Loading model.h5...")
    model = tf.keras.models.load_model('model.h5')

    # ── Keras baseline ────────────────────────────────────────────────────────
    keras_acc, keras_preds = evaluate_keras(model, X_val, y_val)
    print("Keras model accuracy : {:.1f}%  ({}/{})".format(
        keras_acc * 100, int(keras_acc * len(y_val)), len(y_val)))

    # ── Float TFLite ──────────────────────────────────────────────────────────
    print("\nConverting to float32 TFLite...")
    float_bytes = convert_float(model)
    float_preds = run_tflite(float_bytes, X_val)
    float_acc   = np.mean(float_preds == y_val)
    print("Float TFLite accuracy: {:.1f}%  ({}/{})".format(
        float_acc * 100, int(float_acc * len(y_val)), len(y_val)))

    # ── Int8 quantized TFLite ─────────────────────────────────────────────────
    print("\nConverting to int8 quantized TFLite (using 50 calibration images)...")
    quant_bytes = convert_int8(model, X)
    quant_preds = run_tflite(quant_bytes, X_val)
    quant_acc   = np.mean(quant_preds == y_val)
    print("Int8  TFLite accuracy: {:.1f}%  ({}/{})".format(
        quant_acc * 100, int(quant_acc * len(y_val)), len(y_val)))

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Model comparison (val set, n={})".format(len(y_val)))
    print("  Keras model  : {:.1f}%".format(keras_acc * 100))
    print("  Float TFLite : {:.1f}%".format(float_acc * 100))
    print("  Int8  TFLite : {:.1f}%".format(quant_acc * 100))

    match = np.mean(quant_preds == keras_preds)
    print("\nPrediction agreement (int8 vs Keras): {:.1f}%".format(match * 100))
    if match >= 0.95:
        print("Results match: YES — safe to deploy int8 model")
    else:
        print("Results match: MARGINAL — consider using float TFLite instead")

    print("\nFiles for ESP32 deployment:")
    print("  model_quant.tflite  — primary (int8, smaller)")
    print("  model_float.tflite  — fallback (float32, larger but exact match)")
    print("\nNext step: run export_weights.py to generate weights_esp32.py")


if __name__ == '__main__':
    main()

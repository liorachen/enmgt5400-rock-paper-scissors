# export_weights.py — Export BN-fused int8 weights for MicroPython inference
# ENMGT 5400
#
# Loads model.h5, folds BatchNorm into Conv weights, quantizes each layer to
# int8, and writes weights.bin — a compact binary file (~347 KB) to copy to ESP32.
# Scales are printed and hardcoded in inference_esp32.py.
#
# BN fusion eliminates the BatchNorm operations entirely, so inference_esp32.py
# only needs to do Conv → ReLU → MaxPool → Dense.
#
# Requirements: pip install tensorflow numpy
#
# Usage:
#   python export_weights.py

import numpy as np
import tensorflow as tf
import struct
import os

MODEL_PATH  = 'model.h5'
OUTPUT_PATH = 'weights.bin'
CLASSES     = ['rock', 'paper', 'scissors']


def get_layer(model, name):
    """Get layer by partial name match."""
    for layer in model.layers:
        if name in layer.name:
            return layer
    raise ValueError("Layer '{}' not found in model".format(name))


def fuse_bn(conv_kernel, conv_bias, bn_gamma, bn_beta, bn_mean, bn_var, bn_eps=1e-3):
    """
    Fold BatchNorm into Conv weights.

    After fusion, the forward pass is just:
        out = conv(inp, fused_kernel) + fused_bias
        out = relu(out)

    Math:
        y = gamma * (conv(x,W)+b - mean) / sqrt(var+eps) + beta
          = conv(x, W * gamma/std) + (b - mean)*gamma/std + beta
    """
    std = np.sqrt(bn_var + bn_eps)
    scale = bn_gamma / std  # per-channel scale, shape: (n_filters,)

    # W shape: (kH, kW, C_in, n_filters)
    # scale along last axis
    fused_kernel = conv_kernel * scale[np.newaxis, np.newaxis, np.newaxis, :]
    fused_bias   = (conv_bias - bn_mean) * scale + bn_beta

    return fused_kernel, fused_bias


def quantize_to_int8(arr):
    """Scale array to int8 range [-127, 127]. Returns (int8 array, scale)."""
    abs_max = np.abs(arr).max()
    if abs_max == 0:
        return arr.astype(np.int8), 1.0
    scale = abs_max / 127.0
    quantized = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return quantized, float(scale)


def bytes_literal(arr_int8):
    """Convert int8 numpy array to Python bytes literal string."""
    raw = arr_int8.astype(np.int8).tobytes()
    # Escape as hex — safe for all byte values
    return "b'" + ''.join('\\x{:02x}'.format(b & 0xFF) for b in raw) + "'"


def main():
    print("Loading", MODEL_PATH, "...")
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()

    layers_data = []

    # ── Conv+BN blocks (3 blocks) ─────────────────────────────────────────────
    for i in range(1, 4):
        conv_name = 'conv2d' if i == 1 else 'conv2d_{}'.format(i - 1)
        bn_name   = 'batch_normalization' if i == 1 else 'batch_normalization_{}'.format(i - 1)

        conv_layer = get_layer(model, conv_name)
        bn_layer   = get_layer(model, bn_name)

        W, b        = conv_layer.get_weights()            # kernel, bias
        gamma, beta, mean, var = bn_layer.get_weights()   # γ, β, μ, σ²

        fused_W, fused_b = fuse_bn(W, b, gamma, beta, mean, var)

        # Transpose [kH, kW, C_in, n_f] → [n_f, kH, kW, C_in]
        # so conv_maxpool can access weights as fw[f*kH*kW*C_in + ...]
        fused_W_t = fused_W.transpose(3, 0, 1, 2)

        W_q, W_scale = quantize_to_int8(fused_W_t)
        b_q, b_scale = quantize_to_int8(fused_b)

        print("Conv{}: kernel {} → transposed {} → int8, scale={:.6f}".format(
            i, fused_W.shape, fused_W_t.shape, W_scale))

        layers_data.append({
            'name':    'CONV{}'.format(i),
            'W':       W_q,
            'b':       b_q,
            'W_scale': W_scale,
            'b_scale': b_scale,
            'W_shape': list(fused_W_t.shape),  # (n_filters, kH, kW, C_in)
            'b_shape': list(fused_b.shape),
        })

    # ── Dense layers (no BN) ──────────────────────────────────────────────────
    dense1 = get_layer(model, 'dense')
    dense2 = get_layer(model, 'dense_1')

    for name, layer in [('DENSE1', dense1), ('DENSE2', dense2)]:
        W, b = layer.get_weights()
        W_q, W_scale = quantize_to_int8(W)
        b_q, b_scale = quantize_to_int8(b)
        print("{}: weights {} → int8, scale={:.6f}".format(name, W.shape, W_scale))
        layers_data.append({
            'name':    name,
            'W':       W_q,
            'b':       b_q,
            'W_scale': W_scale,
            'b_scale': b_scale,
            'W_shape': list(W.shape),
            'b_shape': list(b.shape),
        })

    # ── Write weights.bin (compact binary, ~347 KB) ───────────────────────────
    # Format: for each layer in order, write W bytes then b bytes (raw int8).
    # Scales are printed below and hardcoded in inference_esp32.py.
    print("\nWriting", OUTPUT_PATH, "...")
    total_bytes = 0

    with open(OUTPUT_PATH, 'wb') as f:
        for ld in layers_data:
            W_bytes = ld['W'].astype(np.int8).tobytes()
            b_bytes = ld['b'].astype(np.int8).tobytes()
            f.write(W_bytes)
            f.write(b_bytes)
            total_bytes += len(W_bytes) + len(b_bytes)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print("Done. {} written ({:.1f} KB)".format(OUTPUT_PATH, size_kb))

    # Print scales — these are hardcoded in inference_esp32.py
    print("\nLayer scales (hardcoded in inference_esp32.py):")
    for ld in layers_data:
        print("  {:7s}  W_scale={:.8f}  B_scale={:.8f}".format(
            ld['name'], ld['W_scale'], ld['b_scale']))

    print("\nCopy weights.bin to your ESP32 flash, then run inference_esp32.py")
    print("Transfer time at 115200 baud: ~{:.0f} seconds".format(total_bytes / 10000))


if __name__ == '__main__':
    main()

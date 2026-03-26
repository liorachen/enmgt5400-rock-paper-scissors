# check_firmware.py — Run this on the ESP32 BEFORE inference_esp32.py
# ENMGT 5400
#
# Copy to ESP32 and run via Thonny / mpremote / ampy.
# Reports what libraries are available and how much RAM is free.
#
# Usage (from Thonny): open file → Run
# Usage (mpremote):    mpremote run check_firmware.py

import gc
import sys

gc.collect()

print("=" * 40)
print("ESP32S3 Firmware Check")
print("=" * 40)
print("MicroPython:", sys.version)
print()

# Check ulab (NumPy-like, fast matrix ops)
try:
    import ulab
    print("ulab     : YES  version =", ulab.__version__)
    HAS_ULAB = True
except ImportError:
    print("ulab     : NO")
    HAS_ULAB = False

# Check tflite (TensorFlow Lite Micro)
try:
    import tflite
    print("tflite   : YES")
    HAS_TFLITE = True
except ImportError:
    print("tflite   : NO")
    HAS_TFLITE = False

# Check camera module
try:
    from camera import Camera
    print("camera   : YES  (cnadler86 driver)")
    HAS_CAMERA = True
except ImportError:
    print("camera   : NO")
    HAS_CAMERA = False

# Check filesystem
try:
    import os
    files = os.listdir('/')
    print("fs files :", files)
except:
    print("fs files : (could not list)")

gc.collect()
print()
print("Free RAM :", gc.mem_free(), "bytes ({:.1f} KB)".format(gc.mem_free() / 1024))
print()

# Recommendation
print("=" * 40)
if HAS_ULAB:
    print("Inference mode: FAST  (ulab matrix ops)")
    print("Estimated inference time: ~1-5 seconds")
elif HAS_TFLITE:
    print("Inference mode: TFLITE")
    print("Estimated inference time: <1 second")
else:
    print("Inference mode: SLOW  (pure Python loops)")
    print("Estimated inference time: 30-120 seconds")
    print("This is correct — it will work, just takes time.")
print("=" * 40)
print()
print("Next: copy weights_esp32.py to ESP32, then run inference_esp32.py")

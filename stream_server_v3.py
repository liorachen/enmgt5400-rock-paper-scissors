# Simple Streaming Server for XIAO ESP32S3 Sense
# For cnadler86 micropython-camera-API firmware
# Modified by: Claude (Anthropic) for Nish's ENMGT 5400 project
#
# Sends raw camera frames over TCP with a simple length header.
# The client handles display/conversion.

import network
import socket as soc
import struct
from camera import Camera
from time import sleep
import gc

# ---- CONFIGURATION ----
WIFI_SSID = '101 Giles street '
WIFI_PASS = 'kitesoccer028'
PORT = 8080
# ------------------------

def connect_wifi(ssid, password):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.disconnect()
    sleep(1)
    print("Scanning for networks...")
    nets = wlan.scan()
    for n in nets:
        print(" Found:", n[0].decode('utf-8', 'ignore'))
    print("Connecting to Wi-Fi:", ssid)
    wlan.connect(ssid, password)
    for i in range(20):
        if wlan.isconnected():
            ip = wlan.ifconfig()[0]
            print("Connected! IP:", ip)
            return ip
        print("Waiting...", i + 1)
        sleep(3)
    print("Wi-Fi connection failed!")
    return None

# Initialize camera (160x120 default — we resize to 32x32 during preprocessing)
cam = Camera()
width = cam.get_pixel_width()
height = cam.get_pixel_height()
fmt = cam.get_pixel_format()
print("Camera: {}x{}, format: {}".format(width, height, fmt))

# Connect Wi-Fi
ip = connect_wifi(WIFI_SSID, WIFI_PASS)
if not ip:
    print("ERROR: No Wi-Fi")
    raise SystemExit

wlan = network.WLAN(network.STA_IF)

# Start TCP server
addr = soc.getaddrinfo('0.0.0.0', PORT)[0][-1]
s = soc.socket(soc.AF_INET, soc.SOCK_STREAM)
s.setsockopt(soc.SOL_SOCKET, soc.SO_REUSEADDR, 1)
s.bind(addr)
s.listen(1)

# Warm up camera — first few captures are slow
print("Warming up camera...")
for _ in range(5):
    cam.capture()
print("Camera ready!")

print("=" * 40)
print("Server ready!")
print("Connect client to: {}:{}".format(ip, PORT))
print("Image: {}x{} format: {}".format(width, height, fmt))
print("=" * 40)

while True:
    # Reconnect WiFi if dropped before accepting a new client
    if not wlan.isconnected():
        print("WiFi lost — reconnecting...")
        ip = connect_wifi(WIFI_SSID, WIFI_PASS)
        if not ip:
            sleep(5)
            continue
        print("Reconnected! New IP:", ip)

    cs, ca = s.accept()
    print("Client connected:", ca)

    # Disable Nagle on client socket to reduce per-frame latency
    try:
        cs.setsockopt(soc.IPPROTO_TCP, soc.TCP_NODELAY, 1)
    except:
        pass

    # Send image dimensions first (width, height, format as 3 unsigned shorts)
    cs.write(struct.pack('>HHH', width, height, fmt))

    frame_count = 0
    try:
        while True:
            # Collect garbage before capture to prevent MicroPython OOM stalls
            if frame_count % 10 == 0:
                gc.collect()

            img = cam.capture()
            data = bytes(img)
            # Two separate writes avoids concatenating a full frame copy in RAM.
            # (header + data concatenation would triple peak memory usage.)
            cs.write(struct.pack('>I', len(data)))
            cs.write(data)
            del data  # free immediately so GC can reclaim it
            frame_count += 1
            if frame_count % 50 == 0:
                print("Frames sent:", frame_count)
    except Exception as e:
        print("Client disconnected:", e)
        print("Total frames sent:", frame_count)
        try:
            cs.close()
        except:
            pass

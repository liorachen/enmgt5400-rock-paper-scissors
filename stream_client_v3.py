# Simple Streaming Client for XIAO ESP32S3 Sense
# For cnadler86 micropython-camera-API firmware
# Modified by: Claude (Anthropic) for Nish's ENMGT 5400 project
#
# Receives raw RGB565 frames from the Xiao and displays them.
# Also saves frames as images for data collection.
#
# Requirements: pip install opencv-python numpy

import socket
import struct
import numpy as np
import cv2
import sys
import os
import time
import threading

# ---- CONFIGURATION ----
SERVER_IP = '172.20.10.3'
SERVER_PORT = 8080
SAVE_DIR = 'captures'        # Directory to save captured images
CONNECT_RETRIES = 10         # How many times to retry connecting
CONNECT_TIMEOUT = 5          # Seconds per connection attempt
RECV_TIMEOUT = 30            # Seconds to wait for frame data (first frame can be slow)
# ------------------------

def rgb565_to_bgr(data, width, height):
    """Convert RGB565 raw bytes to BGR numpy array for OpenCV."""
    # Big-endian uint16 handles byte order in one step
    pixels = np.frombuffer(data, dtype='>u2').astype(np.uint16)
    r = (((pixels >> 11) & 0x1F) * 255 // 31).astype(np.uint8).reshape(height, width)
    g = (((pixels >> 5)  & 0x3F) * 255 // 63).astype(np.uint8).reshape(height, width)
    b = (( pixels        & 0x1F) * 255 // 31).astype(np.uint8).reshape(height, width)
    return np.stack([b, g, r], axis=2)

def recv_exact(sock, n):
    """Receive exactly n bytes — O(n) using bytearray + recv_into (avoids O(n²) string concat)."""
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    while pos < n:
        received = sock.recv_into(view[pos:], n - pos)
        if not received:
            raise ConnectionError("Connection closed")
        pos += received
    return bytes(buf)

# Shared state between receiver thread and display (main) thread
latest_frame = None
frame_lock = threading.Lock()
recv_error: list = [None]   # list so receiver thread can write to it
recv_frame_count = [0]

def receiver_thread(sock, expected_size, width, height):
    """Runs in background — receives frames and updates latest_frame."""
    global latest_frame
    try:
        while True:
            len_data = recv_exact(sock, 4)
            frame_len = struct.unpack('>I', len_data)[0]
            frame_data = recv_exact(sock, frame_len)
            recv_frame_count[0] += 1

            if frame_len == expected_size:
                frame = rgb565_to_bgr(frame_data, width, height)
            else:
                # Might be JPEG after all
                nparr = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                with frame_lock:
                    latest_frame = frame   # always keep only the newest frame
    except Exception as e:
        recv_error[0] = e

def main():
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Connecting to {}:{}...".format(SERVER_IP, SERVER_PORT))

    sock = None
    for attempt in range(1, CONNECT_RETRIES + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(CONNECT_TIMEOUT)
            sock.connect((SERVER_IP, SERVER_PORT))
            break
        except Exception as e:
            sock.close()
            sock = None
            print("Attempt {}/{} failed: {} — retrying in 2s...".format(attempt, CONNECT_RETRIES, e))
            time.sleep(2)

    if sock is None:
        print("Could not connect after {} attempts.".format(CONNECT_RETRIES))
        print("Make sure:")
        print("  1. The streaming server is running on the Xiao")
        print("  2. Your laptop is on the same Wi-Fi network")
        print("  3. The SERVER_IP matches what the server printed")
        sys.exit(1)

    sock.settimeout(RECV_TIMEOUT)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("Connected!")

    # Receive image dimensions (width, height, format)
    header = recv_exact(sock, 6)
    width, height, fmt = struct.unpack('>HHH', header)
    print("Image: {}x{}, format: {}".format(width, height, fmt))

    expected_size = width * height * 2  # RGB565 = 2 bytes per pixel
    print("Expected frame size: {} bytes".format(expected_size))
    print("")
    print("Controls:")
    print("  q = quit")
    print("  s = save current frame")
    print("  r = save as 'rock'")
    print("  p = save as 'paper'")
    print("  x = save as 'scissors'")
    print("")

    save_counts = {'rock': 0, 'paper': 0, 'scissors': 0, 'misc': 0}
    display_frame_count = 0

    # Create class directories
    for cls in ['rock', 'paper', 'scissors']:
        os.makedirs(os.path.join(SAVE_DIR, cls), exist_ok=True)

    # Start background receiver thread — decouples network I/O from display
    t = threading.Thread(target=receiver_thread,
                         args=(sock, expected_size, width, height),
                         daemon=True)
    t.start()
    print("Waiting for first frame...")

    try:
        while True:
            # If receiver thread died, report and exit
            if recv_error[0] is not None:
                print("Stream error:", recv_error[0])
                break

            with frame_lock:
                frame = latest_frame

            if frame is None:
                cv2.waitKey(10)   # no frame yet — yield to OS
                continue

            display_frame_count += 1

            # Resize for better viewing
            display = cv2.resize(frame, (width * 3, height * 3),
                                 interpolation=cv2.INTER_NEAREST)

            # Add info overlay
            info = "RX:{} DISP:{} | r=rock p=paper x=scissors s=save q=quit".format(
                recv_frame_count[0], display_frame_count)
            cv2.putText(display, info, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            counts = "Rock:{} Paper:{} Scissors:{}".format(
                save_counts['rock'], save_counts['paper'], save_counts['scissors'])
            cv2.putText(display, counts, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            cv2.imshow('XIAO Camera - RPS Data Collection', display)

            # 30ms wait = ~33fps display cap; receiver thread runs independently
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                save_counts['misc'] += 1
                fn = os.path.join(SAVE_DIR, 'frame_{:04d}.bmp'.format(save_counts['misc']))
                cv2.imwrite(fn, frame)
                print("Saved:", fn)
            elif key == ord('r'):
                save_counts['rock'] += 1
                fn = os.path.join(SAVE_DIR, 'rock', 'rock_{:04d}.bmp'.format(save_counts['rock']))
                cv2.imwrite(fn, frame)
                print("Saved:", fn, "({} total rock)".format(save_counts['rock']))
            elif key == ord('p'):
                save_counts['paper'] += 1
                fn = os.path.join(SAVE_DIR, 'paper', 'paper_{:04d}.bmp'.format(save_counts['paper']))
                cv2.imwrite(fn, frame)
                print("Saved:", fn, "({} total paper)".format(save_counts['paper']))
            elif key == ord('x'):
                save_counts['scissors'] += 1
                fn = os.path.join(SAVE_DIR, 'scissors', 'scissors_{:04d}.bmp'.format(save_counts['scissors']))
                cv2.imwrite(fn, frame)
                print("Saved:", fn, "({} total scissors)".format(save_counts['scissors']))

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print("Error:", e)

    if sock is not None:
        sock.close()
    cv2.destroyAllWindows()

    print("\n=== Collection Summary ===")
    print("Frames received by network:", recv_frame_count[0])
    print("Frames displayed:", display_frame_count)
    for cls, count in save_counts.items():
        if count > 0:
            print("  {}: {} images".format(cls, count))

if __name__ == '__main__':
    main()

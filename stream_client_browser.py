# Browser-Based Streaming Client for XIAO ESP32S3 Sense
# Opens a live view in your web browser instead of OpenCV
# Modified by: Claude (Anthropic) for Nish's ENMGT 5400 project
#
# Requirements: pip install numpy Pillow
# Usage: python3 stream_client_browser.py
#        Then open http://localhost:9090 in your browser

import socket
import struct
import numpy as np
from PIL import Image
import io
import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import webbrowser

# ---- CONFIGURATION ----
SERVER_IP = '192.168.4.40'   # Change to match server output
SERVER_PORT = 8080
WEB_PORT = 9090               # Local web server port
SAVE_DIR = 'captures'        # Directory to save captured images
# ------------------------

# Global state
latest_frame_jpg = None
frame_lock = threading.Lock()   # guards both latest_frame_jpg and latest_raw_frame
frame_count = 0
save_counts = {'rock': 0, 'paper': 0, 'scissors': 0}
status_msg = "Connecting..."

def rgb565_to_image(data, width, height):
    """Convert RGB565 raw bytes to PIL Image."""
    pixels = np.frombuffer(data, dtype='>u2')
    r = ((pixels >> 11) & 0x1F).astype(np.uint8)
    g = ((pixels >> 5) & 0x3F).astype(np.uint8)
    b = (pixels & 0x1F).astype(np.uint8)
    r = (r << 3) | (r >> 2)
    g = (g << 2) | (g >> 4)
    b = (b << 3) | (b >> 2)
    rgb = np.stack([r, g, b], axis=1).reshape(height, width, 3)
    return Image.fromarray(rgb, 'RGB')

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

def save_image(category, frame_data, width, height):
    """Save a frame as BMP in the appropriate category folder."""
    global save_counts
    save_counts[category] += 1
    folder = os.path.join(SAVE_DIR, category)
    os.makedirs(folder, exist_ok=True)
    fn = os.path.join(folder, '{}_{:04d}.bmp'.format(category, save_counts[category]))
    img = rgb565_to_image(frame_data, width, height)
    img.save(fn, 'BMP')
    return fn

# Store latest raw frame for saving (protected by frame_lock)
latest_raw_frame = None
img_width = 0
img_height = 0

class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, _format, *_args):
        pass  # Suppress log output

    def do_GET(self):
        global latest_raw_frame, save_counts, status_msg

        # Strip query string for path matching (fixes ?timestamp cache busting)
        path = urlparse(self.path).path

        if path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())

        elif path == '/frame.jpg':
            with frame_lock:
                jpg = latest_frame_jpg
            if jpg:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(jpg)))
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.end_headers()
                self.wfile.write(jpg)
            else:
                # Return 1x1 black JPEG placeholder
                placeholder = create_placeholder_jpeg()
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(placeholder)))
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(placeholder)

        elif path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            import json
            self.wfile.write(json.dumps({
                'frame_count': frame_count,
                'rock': save_counts['rock'],
                'paper': save_counts['paper'],
                'scissors': save_counts['scissors'],
                'status': status_msg
            }).encode())

        elif path.startswith('/save/'):
            category = path.split('/save/')[1]
            with frame_lock:
                raw = latest_raw_frame
                w, h = img_width, img_height
            if category in ['rock', 'paper', 'scissors'] and raw is not None:
                fn = save_image(category, raw, w, h)
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                msg = "Saved {} #{} ({})".format(category, save_counts[category], fn)
                print(msg)
                self.wfile.write(msg.encode())
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'No frame available or invalid category')

        else:
            self.send_response(404)
            self.end_headers()

def create_placeholder_jpeg():
    """Create a small black placeholder JPEG."""
    img = Image.new('RGB', (160, 120), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=50)
    return buf.getvalue()

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>XIAO Camera - RPS Data Collection</title>
<style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        background: #1a1a2e; color: #eee; text-align: center;
        margin: 0; padding: 20px;
    }
    h1 { color: #e94560; margin-bottom: 5px; }
    .subtitle { color: #888; margin-bottom: 20px; }
    #cam {
        image-rendering: pixelated;
        width: 480px; height: 360px;
        border: 3px solid #333; border-radius: 8px;
        background: #000;
    }
    .controls { margin: 20px 0; }
    .btn {
        font-size: 18px; padding: 12px 30px; margin: 5px 10px;
        border: none; border-radius: 8px; cursor: pointer;
        color: white; font-weight: bold;
        transition: transform 0.1s;
    }
    .btn:hover { transform: scale(1.05); }
    .btn:active { transform: scale(0.95); }
    .btn-rock { background: #e94560; }
    .btn-paper { background: #0f3460; }
    .btn-scissors { background: #533483; }
    .stats {
        display: flex; justify-content: center; gap: 30px;
        margin: 15px 0; font-size: 16px;
    }
    .stat {
        padding: 8px 20px; border-radius: 6px;
        background: #16213e; min-width: 100px;
    }
    .stat-label { color: #888; font-size: 12px; }
    .stat-value { font-size: 24px; font-weight: bold; }
    .info { color: #666; font-size: 13px; margin-top: 15px; }
    .status { color: #4ecca3; margin: 10px 0; }
    .keyboard-hint {
        color: #666; font-size: 12px; margin-top: 5px;
        background: #16213e; display: inline-block;
        padding: 4px 10px; border-radius: 4px;
    }
    .saved-msg {
        color: #4ecca3; font-size: 14px; min-height: 20px;
        margin: 10px 0;
    }
</style>
</head>
<body>
<h1>XIAO ESP32S3 Camera</h1>
<p class="subtitle">Rock Paper Scissors Data Collection</p>
<div class="status" id="status">Connecting...</div>
<img id="cam" src="/frame.jpg">
<div class="saved-msg" id="savedMsg"></div>
<div class="controls">
    <button class="btn btn-rock" onclick="save('rock')">&#9994; Rock (R)</button>
    <button class="btn btn-paper" onclick="save('paper')">&#9995; Paper (P)</button>
    <button class="btn btn-scissors" onclick="save('scissors')">&#9996; Scissors (X)</button>
</div>
<div class="keyboard-hint">Keyboard shortcuts: R = Rock, P = Paper, X = Scissors</div>
<div class="stats">
    <div class="stat">
        <div class="stat-label">ROCK</div>
        <div class="stat-value" id="rock">0</div>
    </div>
    <div class="stat">
        <div class="stat-label">PAPER</div>
        <div class="stat-value" id="paper">0</div>
    </div>
    <div class="stat">
        <div class="stat-label">SCISSORS</div>
        <div class="stat-value" id="scissors">0</div>
    </div>
    <div class="stat">
        <div class="stat-label">FRAMES</div>
        <div class="stat-value" id="frames">0</div>
    </div>
</div>
<p class="info">Images save to captures/rock/, captures/paper/, captures/scissors/ as BMP files</p>
<script>
    function refreshImage() {
        var img = document.getElementById('cam');
        var newImg = new Image();
        newImg.onload = function() {
            img.src = newImg.src;
        };
        newImg.src = '/frame.jpg?' + Date.now();
    }
    setInterval(refreshImage, 200);

    function updateStats() {
        fetch('/status')
            .then(function(r) { return r.json(); })
            .then(function(d) {
                document.getElementById('rock').textContent = d.rock;
                document.getElementById('paper').textContent = d.paper;
                document.getElementById('scissors').textContent = d.scissors;
                document.getElementById('frames').textContent = d.frame_count;
                document.getElementById('status').textContent = d.status;
            })
            .catch(function() {});
    }
    setInterval(updateStats, 500);

    function save(category) {
        fetch('/save/' + category)
            .then(function(r) { return r.text(); })
            .then(function(msg) {
                document.getElementById('savedMsg').textContent = msg;
                updateStats();
                setTimeout(function() {
                    document.getElementById('savedMsg').textContent = '';
                }, 2000);
            })
            .catch(function(err) {
                document.getElementById('savedMsg').textContent = 'Error: ' + err;
            });
    }

    document.addEventListener('keydown', function(e) {
        if (e.key === 'r' || e.key === 'R') save('rock');
        if (e.key === 'p' || e.key === 'P') save('paper');
        if (e.key === 'x' || e.key === 'X') save('scissors');
    });
</script>
</body>
</html>"""

def camera_thread():
    """Connect to Xiao and receive frames."""
    global latest_frame_jpg, latest_raw_frame, frame_count, img_width, img_height, status_msg

    while True:
        try:
            print("Connecting to {}:{}...".format(SERVER_IP, SERVER_PORT))
            status_msg = "Connecting to Xiao..."

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((SERVER_IP, SERVER_PORT))
            sock.settimeout(30)  # 30s timeout after connect (first frame can be slow)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            # Receive image dimensions
            header = recv_exact(sock, 6)
            width, height, fmt = struct.unpack('>HHH', header)
            img_width = width
            img_height = height
            print("Connected! Image: {}x{}, format: {}".format(width, height, fmt))
            status_msg = "Streaming {}x{} @ format {}".format(width, height, fmt)

            expected_size = width * height * 2

            while True:
                # Receive frame using fast bytearray path
                len_data = recv_exact(sock, 4)
                frame_len = struct.unpack('>I', len_data)[0]
                frame_data = recv_exact(sock, frame_len)
                frame_count += 1

                # Convert to JPEG for browser display
                try:
                    if frame_len == expected_size:
                        img = rgb565_to_image(frame_data, width, height)
                    else:
                        # JPEG or other compressed format — decode directly
                        img = Image.open(io.BytesIO(frame_data))

                    buf = io.BytesIO()
                    img.save(buf, format='JPEG', quality=80)

                    # Update both jpg and raw frame atomically under one lock
                    with frame_lock:
                        latest_frame_jpg = buf.getvalue()
                        latest_raw_frame = frame_data

                    if frame_count == 1:
                        print("First frame received! ({} bytes JPEG)".format(len(latest_frame_jpg)))

                except Exception as conv_err:
                    print("Frame conversion error:", conv_err)
                    import traceback
                    traceback.print_exc()

                if frame_count % 100 == 0:
                    print("Frames received:", frame_count)

        except Exception as e:
            print("Connection error:", e)
            import traceback
            traceback.print_exc()
            status_msg = "Disconnected. Reconnecting..."
            time.sleep(3)

def main():
    for cls in ['rock', 'paper', 'scissors']:
        os.makedirs(os.path.join(SAVE_DIR, cls), exist_ok=True)

    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()

    url = 'http://localhost:{}'.format(WEB_PORT)
    print("Starting web server on {}".format(url))
    print("Opening browser...")
    # Use webbrowser.open() — works on all platforms without needing Chrome specifically
    webbrowser.open(url)

    print("")
    print("Controls (in browser):")
    print("  Click buttons or press R=Rock, P=Paper, X=Scissors")
    print("  Images save to captures/rock/, captures/paper/, captures/scissors/")
    print("")
    print("Press Ctrl+C in this terminal to stop")

    HTTPServer.allow_reuse_address = True
    server = HTTPServer(('localhost', WEB_PORT), StreamHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
        print("\n=== Collection Summary ===")
        print("Total frames received:", frame_count)
        for cls, count in save_counts.items():
            print("  {}: {} images".format(cls, count))

if __name__ == '__main__':
    main()

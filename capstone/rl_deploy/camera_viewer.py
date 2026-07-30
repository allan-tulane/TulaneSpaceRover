#!/usr/bin/env python3
"""
Camera Viewer — Team Crater
==============================
Live camera feed from the rover's OAK-D Lite.
Shows: RGB + detections, Depth heatmap, Segmentation mask (if enabled).

Access from any browser on the same Tailscale/network:
    http://100.68.244.40:8080

Usage:
    python3 camera_viewer.py                          # Default
    python3 camera_viewer.py --port 9090              # Custom port
    python3 camera_viewer.py --segmentation           # With terrain mask
    python3 camera_viewer.py --seg-model path/to.pth  # Custom model path
"""

import sys
import os
import time
import argparse
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ugv_jetson'))

from flask import Flask, Response, render_template_string
import json as _json

app = Flask(__name__)

# Global state
_frame_lock = threading.Lock()
_latest_rgb = None
_latest_depth = None
_latest_seg = None
_detection_info = ""
_fps = 0

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Team Crater — Rover Camera</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #1B3A2D; color: #e0e0e0;
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            display: flex; flex-direction: column; align-items: center;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #006747 0%, #1B3A2D 100%);
            width: 100%; padding: 18px 0 14px; text-align: center;
            border-bottom: 3px solid #87CEEB;
        }
        h1 { color: #87CEEB; margin: 0 0 4px; font-size: 1.5em; letter-spacing: 1px; }
        .subtitle { color: #9BC4A8; font-size: 0.85em; margin-bottom: 0; }
        .tulane-badge { color: #87CEEB; font-weight: bold; font-size: 0.75em; margin-top: 4px; }
        .streams {
            display: flex; gap: 12px; flex-wrap: wrap; justify-content: center;
            padding: 15px 10px;
        }
        .stream-box {
            background: #0D2818; border: 1px solid #006747;
            border-radius: 8px; overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .stream-box h2 {
            background: #006747; color: #87CEEB;
            padding: 8px 12px; font-size: 0.9em; text-align: center;
            letter-spacing: 0.5px;
        }
        img.feed { display: block; width: 480px; height: auto; }
        .info {
            background: #0D2818; border: 1px solid #006747;
            border-radius: 8px; padding: 12px 20px; margin: 0 15px 15px;
            font-size: 0.85em; max-width: 1500px; width: 95%;
            font-family: 'Courier New', monospace;
        }
        .info span { color: #87CEEB; }
        .info .label { color: #9BC4A8; }
        #status { color: #87CEEB; }
        .legend { display: flex; gap: 15px; margin-top: 8px; flex-wrap: wrap; }
        .legend-item { display: flex; align-items: center; gap: 5px; }
        .legend-swatch { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #555; }
        .footer { color: #555; font-size: 0.7em; margin: 10px 0 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Team Crater — Rover Camera</h1>
        <p class="subtitle">OAK-D Lite | Live Feed | Face/Person/Object Detection</p>
        <p class="tulane-badge">TULANE UNIVERSITY | CMPS 4020 CAPSTONE | SPRING 2026</p>
    </div>
    <div class="streams">
        <div class="stream-box">
            <h2>RGB + Detections</h2>
            <img class="feed" src="/stream/rgb" alt="RGB" />
        </div>
        <div class="stream-box">
            <h2>Depth Map</h2>
            <img class="feed" src="/stream/depth" alt="Depth" />
        </div>
        {% if segmentation %}
        <div class="stream-box">
            <h2>Terrain Segmentation</h2>
            <img class="feed" src="/stream/seg" alt="Segmentation" />
        </div>
        {% endif %}
    </div>
    <div class="info">
        <p>Status: <span id="status">Connecting...</span></p>
        <p>Detection: <span id="det_info">Loading...</span></p>
        <p>Objects: <span id="obj_info">-</span></p>
        {% if segmentation %}
        <div class="legend" style="margin-top:8px;">
            <span style="color:#888;">Segmentation:</span>
            <div class="legend-item"><div class="legend-swatch" style="background:#333;"></div> Ground</div>
            <div class="legend-item"><div class="legend-swatch" style="background:#4444ff;"></div> Sky</div>
            <div class="legend-item"><div class="legend-swatch" style="background:#00cc00;"></div> Small Rock</div>
            <div class="legend-item"><div class="legend-swatch" style="background:#ff0000;"></div> Big Rock</div>
        </div>
        {% endif %}
    </div>
    <p class="footer">Mary Ella Petersen | Arzaan Irani | Mark Tikhonov | Advisor: Dr. Jihun Hamm</p>
    <script>
        setInterval(async () => {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                document.getElementById('det_info').textContent = data.detection;
                document.getElementById('obj_info').textContent = data.objects;
                document.getElementById('status').textContent = 'Connected (' + data.fps + ' fps)';
                document.getElementById('status').style.color = '#00ff88';
            } catch(e) {
                document.getElementById('status').textContent = 'Disconnected';
                document.getElementById('status').style.color = '#ff6b6b';
            }
        }, 1000);
    </script>
</body>
</html>
"""


def camera_thread(enable_seg, seg_model_path):
    """Background thread: captures frames, runs detection + segmentation."""
    global _latest_rgb, _latest_depth, _latest_seg, _detection_info, _fps
    import cv2
    from rl_deploy import DepthCamera, FaceTracker, ObjectDetector

    cam = DepthCamera(enable_rgb=True)
    time.sleep(1)

    # Object detector (MobileNet-SSD)
    try:
        obj_detector = ObjectDetector()
    except Exception as e:
        print(f"[WARN] Object detector unavailable: {e}")
        obj_detector = None

    # Face tracker (DNN + Haar + Person)
    face_tracker = FaceTracker(detector=obj_detector)

    # Segmentation
    segmenter = None
    if enable_seg and seg_model_path:
        try:
            from rl_deploy import TerrainSegmenter
            segmenter = TerrainSegmenter(seg_model_path)
            print("[OK] Terrain segmenter loaded for viewer")
        except Exception as e:
            print(f"[WARN] Segmenter unavailable: {e}")

    # Seg colormap: ground=dark, sky=blue, rock=green, hill=red
    seg_colors = np.array([
        [0, 0, 255],     # class 0 - hills/mountain 
        [0, 200, 0],    # class 1 - rock
        [50, 50, 50],    # class 2 - lunar regolith
        [200, 0, 120],     # class 3 - crater
        [255, 80, 80],    # class 4 - sky
    ], dtype=np.uint8)

    fps_counter = 0
    fps_time = time.time()
    obj_info_str = ""
    seg_step = 0

    while True:
        try:
            depth_frame, min_depth, left_depth, right_depth = cam.get_depth_frame()
            rgb = cam.get_rgb_frame()

            if rgb is None:
                time.sleep(0.05)
                continue

            overlay = rgb.copy()

            # --- Face/person detection ---
            found, angle, dist = face_tracker.detect(rgb, depth_frame)
            if found:
                method = getattr(face_tracker, '_last_method', '?')
                det_text = f"TARGET: {method} angle={np.degrees(angle):.1f} dist={dist:.2f}m"
                h, w = overlay.shape[:2]
                cx = int((angle / np.radians(73) + 0.5) * w)
                cx = max(0, min(w - 1, cx))
                # Draw targeting reticle
                cv2.circle(overlay, (cx, h // 3), 35, (0, 255, 0), 3)
                cv2.line(overlay, (cx - 45, h // 3), (cx + 45, h // 3), (0, 255, 0), 2)
                cv2.line(overlay, (cx, h // 3 - 45), (cx, h // 3 + 45), (0, 255, 0), 2)
                cv2.putText(overlay, f"{method} {dist:.1f}m",
                            (max(5, cx - 70), h // 3 - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                rate = 0
                if face_tracker._detect_count > 0:
                    rate = face_tracker._found_count / face_tracker._detect_count * 100
                det_text = f"Searching... ({face_tracker._found_count}/{face_tracker._detect_count} = {rate:.0f}%)"

            # --- MobileNet object detection overlay ---
            obj_texts = []
            if obj_detector is not None:
                try:
                    detections = obj_detector.detect(rgb)
                    h, w = overlay.shape[:2]
                    for det in detections:
                        cls = det['class']
                        conf = det['confidence']
                        bbox = det['bbox']
                        x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
                        x2, y2 = int(bbox[2] * w), int(bbox[3] * h)
                        # Color: green for person, yellow for others
                        color = (0, 255, 0) if cls == 'person' else (0, 255, 255)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                        label = f"{cls} {conf:.0%}"
                        cv2.putText(overlay, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        obj_texts.append(f"{cls}({conf:.0%})")
                except Exception:
                    pass
            obj_info_str = ", ".join(obj_texts) if obj_texts else "None"

            # Depth text overlay
            cv2.putText(overlay, f"Depth: min={min_depth:.2f}m L={left_depth:.2f} R={right_depth:.2f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # --- Depth visualization ---
            depth_vis = np.clip(depth_frame / 5.0, 0, 1)
            depth_vis = (depth_vis * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            depth_color = cv2.resize(depth_color, (640, 480))

            # --- Segmentation (every 3rd frame to save CPU) ---
            seg_vis = None
            if segmenter is not None and seg_step % 3 == 0:
                try:
                    overall, left_f, right_f = segmenter.analyze(rgb)
                    # Get the mask from the last inference
                    if hasattr(segmenter, '_last_mask') and segmenter._last_mask is not None:
                        mask = segmenter._last_mask
                    else:
                        # Generate mask from model
                        import torch
                        img_resized = cv2.resize(rgb, (720, 480))
                        img_norm = img_resized.astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)
                        if hasattr(segmenter, 'model'):
                            with torch.no_grad():
                                pred = segmenter.model(img_tensor)
                            mask = pred.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
                        else:
                            mask = np.zeros((480, 720), dtype=np.uint8)
                    # Colorize
                    seg_vis = seg_colors[mask]
                    # Blend with original for visibility
                    rgb_resized = cv2.resize(rgb, (mask.shape[1], mask.shape[0]))
                    seg_vis = cv2.addWeighted(rgb_resized, 0.4, seg_vis, 0.6, 0)
                    seg_vis = cv2.resize(seg_vis, (640, 480))
                    cv2.putText(seg_vis, f"Feasibility: L={left_f:.2f} R={right_f:.2f} All={overall:.2f}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as e:
                    seg_vis = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(seg_vis, f"Seg error: {str(e)[:50]}", (10, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            seg_step += 1

            # FPS
            fps_counter += 1
            if time.time() - fps_time > 1.0:
                _fps = fps_counter
                fps_counter = 0
                fps_time = time.time()

            with _frame_lock:
                _latest_rgb = overlay
                _latest_depth = depth_color
                if seg_vis is not None:
                    _latest_seg = seg_vis
                _detection_info = det_text
                app.config['OBJ_INFO'] = obj_info_str

        except Exception as e:
            print(f"[VIEWER] Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.5)


def gen_stream(frame_type):
    """Generator for MJPEG stream."""
    import cv2
    while True:
        with _frame_lock:
            if frame_type == 'rgb':
                frame = _latest_rgb
            elif frame_type == 'depth':
                frame = _latest_depth
            elif frame_type == 'seg':
                frame = _latest_seg
            else:
                frame = None

        if frame is None:
            time.sleep(0.1)
            continue

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.05)


@app.route('/')
def index():
    return render_template_string(HTML_PAGE,
                                  port=app.config.get('PORT', 8080),
                                  segmentation=app.config.get('SEG', False))


@app.route('/stream/rgb')
def stream_rgb():
    return Response(gen_stream('rgb'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream/depth')
def stream_depth():
    return Response(gen_stream('depth'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream/seg')
def stream_seg():
    return Response(gen_stream('seg'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    return _json.dumps({
        'detection': _detection_info or 'No data',
        'objects': app.config.get('OBJ_INFO', '-'),
        'fps': _fps,
    }), 200, {'Content-Type': 'application/json'}


def main():
    parser = argparse.ArgumentParser(description='Rover Camera Viewer')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--segmentation', action='store_true',
                        help='Enable terrain segmentation mask view')
    parser.add_argument('--seg-model', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'mark_model', 'unet_lunar_segmentation.pth'))
    args = parser.parse_args()

    app.config['PORT'] = args.port
    app.config['SEG'] = args.segmentation

    t = threading.Thread(target=camera_thread,
                         args=(args.segmentation, args.seg_model),
                         daemon=True)
    t.start()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  Team Crater — Camera Viewer                             ║
║  Open in browser: http://0.0.0.0:{args.port:<5}                ║
║  Detection: DNN Face + Haar + MobileNet Person/Objects   ║
║  Segmentation: {'ON' if args.segmentation else 'OFF':42} ║
╚══════════════════════════════════════════════════════════╝
""")

    app.run(host='0.0.0.0', port=args.port, threaded=True)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Camera Viewer — Team Crater
==============================
Live camera feed from the rover's OAK-D Lite.
Shows: RGB + detections, Depth heatmap, Segmentation mask (if enabled).

Access from any browser on the same Tailscale/network:
    http://100.68.244.40:8080

Usage:
    python3 camera_viewer.py                          # Default
    python3 camera_viewer.py --port 9090              # Custom port
    python3 camera_viewer.py --segmentation           # With terrain mask
    python3 camera_viewer.py --seg-model path/to.pth  # Custom model path
"""

import sys
import os
import time
import argparse
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ugv_jetson'))

from flask import Flask, Response, render_template_string
import json as _json

app = Flask(__name__)

# Global state
_frame_lock = threading.Lock()
_latest_rgb = None
_latest_depth = None
_latest_seg = None
_detection_info = ""
_fps = 0

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Team Crater — Rover Camera</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #1B3A2D; color: #e0e0e0;
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            display: flex; flex-direction: column; align-items: center;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #006747 0%, #1B3A2D 100%);
            width: 100%; padding: 18px 0 14px; text-align: center;
            border-bottom: 3px solid #87CEEB;
        }
        h1 { color: #87CEEB; margin: 0 0 4px; font-size: 1.5em; letter-spacing: 1px; }
        .subtitle { color: #9BC4A8; font-size: 0.85em; margin-bottom: 0; }
        .tulane-badge { color: #87CEEB; font-weight: bold; font-size: 0.75em; margin-top: 4px; }
        .streams {
            display: flex; gap: 12px; flex-wrap: wrap; justify-content: center;
            padding: 15px 10px;
        }
        .stream-box {
            background: #0D2818; border: 1px solid #006747;
            border-radius: 8px; overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .stream-box h2 {
            background: #006747; color: #87CEEB;
            padding: 8px 12px; font-size: 0.9em; text-align: center;
            letter-spacing: 0.5px;
        }
        img.feed { display: block; width: 480px; height: auto; }
        .info {
            background: #0D2818; border: 1px solid #006747;
            border-radius: 8px; padding: 12px 20px; margin: 0 15px 15px;
            font-size: 0.85em; max-width: 1500px; width: 95%;
            font-family: 'Courier New', monospace;
        }
        .info span { color: #87CEEB; }
        .info .label { color: #9BC4A8; }
        #status { color: #87CEEB; }
        .legend { display: flex; gap: 15px; margin-top: 8px; flex-wrap: wrap; }
        .legend-item { display: flex; align-items: center; gap: 5px; }
        .legend-swatch { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #555; }
        .footer { color: #555; font-size: 0.7em; margin: 10px 0 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Team Crater — Rover Camera</h1>
        <p class="subtitle">OAK-D Lite | Live Feed | Face/Person/Object Detection</p>
        <p class="tulane-badge">TULANE UNIVERSITY | CMPS 4020 CAPSTONE | SPRING 2026</p>
    </div>
    <div class="streams">
        <div class="stream-box">
            <h2>RGB + Detections</h2>
            <img class="feed" src="/stream/rgb" alt="RGB" />
        </div>
        <div class="stream-box">
            <h2>Depth Map</h2>
            <img class="feed" src="/stream/depth" alt="Depth" />
        </div>
        {% if segmentation %}
        <div class="stream-box">
            <h2>Terrain Segmentation</h2>
            <img class="feed" src="/stream/seg" alt="Segmentation" />
        </div>
        {% endif %}
    </div>
    <div class="info">
        <p>Status: <span id="status">Connecting...</span></p>
        <p>Detection: <span id="det_info">Loading...</span></p>
        <p>Objects: <span id="obj_info">-</span></p>
        {% if segmentation %}
        <div class="legend" style="margin-top:8px;">
            <span style="color:#888;">Segmentation:</span>
            <div class="legend-item"><div class="legend-swatch" style="background:#333;"></div> Ground</div>
            <div class="legend-item"><div class="legend-swatch" style="background:#4444ff;"></div> Sky</div>
            <div class="legend-item"><div class="legend-swatch" style="background:#00cc00;"></div> Small Rock</div>
            <div class="legend-item"><div class="legend-swatch" style="background:#ff0000;"></div> Big Rock</div>
        </div>
        {% endif %}
    </div>
    <p class="footer">Mary Ella Petersen | Arzaan Irani | Mark Tikhonov | Advisor: Dr. Jihun Hamm</p>
    <script>
        setInterval(async () => {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                document.getElementById('det_info').textContent = data.detection;
                document.getElementById('obj_info').textContent = data.objects;
                document.getElementById('status').textContent = 'Connected (' + data.fps + ' fps)';
                document.getElementById('status').style.color = '#00ff88';
            } catch(e) {
                document.getElementById('status').textContent = 'Disconnected';
                document.getElementById('status').style.color = '#ff6b6b';
            }
        }, 1000);
    </script>
</body>
</html>
"""


def camera_thread(enable_seg, seg_model_path):
    """Background thread: captures frames, runs detection + segmentation."""
    global _latest_rgb, _latest_depth, _latest_seg, _detection_info, _fps
    import cv2
    from rl_deploy import DepthCamera, FaceTracker, ObjectDetector

    cam = DepthCamera(enable_rgb=True)
    time.sleep(1)

    # Object detector (MobileNet-SSD)
    try:
        obj_detector = ObjectDetector()
    except Exception as e:
        print(f"[WARN] Object detector unavailable: {e}")
        obj_detector = None

    # Face tracker (DNN + Haar + Person)
    face_tracker = FaceTracker(detector=obj_detector)

    # Segmentation
    segmenter = None
    if enable_seg and seg_model_path:
        try:
            from rl_deploy import TerrainSegmenter
            segmenter = TerrainSegmenter(seg_model_path)
            print("[OK] Terrain segmenter loaded for viewer")
        except Exception as e:
            print(f"[WARN] Segmenter unavailable: {e}")

    # Seg colormap: ground=dark, sky=blue, rock=green, hill=red
    seg_colors = np.array([
        [0, 0, 255],     # class 0 - hills/mountain 
        [0, 200, 0],    # class 1 - rock
        [50, 50, 50],    # class 2 - lunar regolith
        [200, 0, 120],     # class 3 - crater
        [255, 80, 80],    # class 4 - sky
    ], dtype=np.uint8)

    fps_counter = 0
    fps_time = time.time()
    obj_info_str = ""
    seg_step = 0

    while True:
        try:
            depth_frame, min_depth, left_depth, right_depth = cam.get_depth_frame()
            rgb = cam.get_rgb_frame()

            if rgb is None:
                time.sleep(0.05)
                continue

            overlay = rgb.copy()

            # --- Face/person detection ---
            found, angle, dist = face_tracker.detect(rgb, depth_frame)
            if found:
                method = getattr(face_tracker, '_last_method', '?')
                det_text = f"TARGET: {method} angle={np.degrees(angle):.1f} dist={dist:.2f}m"
                h, w = overlay.shape[:2]
                cx = int((angle / np.radians(73) + 0.5) * w)
                cx = max(0, min(w - 1, cx))
                # Draw targeting reticle
                cv2.circle(overlay, (cx, h // 3), 35, (0, 255, 0), 3)
                cv2.line(overlay, (cx - 45, h // 3), (cx + 45, h // 3), (0, 255, 0), 2)
                cv2.line(overlay, (cx, h // 3 - 45), (cx, h // 3 + 45), (0, 255, 0), 2)
                cv2.putText(overlay, f"{method} {dist:.1f}m",
                            (max(5, cx - 70), h // 3 - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                rate = 0
                if face_tracker._detect_count > 0:
                    rate = face_tracker._found_count / face_tracker._detect_count * 100
                det_text = f"Searching... ({face_tracker._found_count}/{face_tracker._detect_count} = {rate:.0f}%)"

            # --- MobileNet object detection overlay ---
            obj_texts = []
            if obj_detector is not None:
                try:
                    detections = obj_detector.detect(rgb)
                    h, w = overlay.shape[:2]
                    for det in detections:
                        cls = det['class']
                        conf = det['confidence']
                        bbox = det['bbox']
                        x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
                        x2, y2 = int(bbox[2] * w), int(bbox[3] * h)
                        # Color: green for person, yellow for others
                        color = (0, 255, 0) if cls == 'person' else (0, 255, 255)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                        label = f"{cls} {conf:.0%}"
                        cv2.putText(overlay, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        obj_texts.append(f"{cls}({conf:.0%})")
                except Exception:
                    pass
            obj_info_str = ", ".join(obj_texts) if obj_texts else "None"

            # Depth text overlay
            cv2.putText(overlay, f"Depth: min={min_depth:.2f}m L={left_depth:.2f} R={right_depth:.2f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # --- Depth visualization ---
            depth_vis = np.clip(depth_frame / 5.0, 0, 1)
            depth_vis = (depth_vis * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            depth_color = cv2.resize(depth_color, (640, 480))

            # --- Segmentation (every 3rd frame to save CPU) ---
            seg_vis = None
            if segmenter is not None and seg_step % 3 == 0:
                try:
                    overall, left_f, right_f = segmenter.analyze(rgb)
                    # Get the mask from the last inference
                    if hasattr(segmenter, '_last_mask') and segmenter._last_mask is not None:
                        mask = segmenter._last_mask
                    else:
                        # Generate mask from model
                        import torch
                        img_resized = cv2.resize(rgb, (720, 480))
                        img_norm = img_resized.astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)
                        if hasattr(segmenter, 'model'):
                            with torch.no_grad():
                                pred = segmenter.model(img_tensor)
                            mask = pred.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
                        else:
                            mask = np.zeros((480, 720), dtype=np.uint8)
                    # Colorize
                    seg_vis = seg_colors[mask]
                    # Blend with original for visibility
                    rgb_resized = cv2.resize(rgb, (mask.shape[1], mask.shape[0]))
                    seg_vis = cv2.addWeighted(rgb_resized, 0.4, seg_vis, 0.6, 0)
                    seg_vis = cv2.resize(seg_vis, (640, 480))
                    cv2.putText(seg_vis, f"Feasibility: L={left_f:.2f} R={right_f:.2f} All={overall:.2f}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as e:
                    seg_vis = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(seg_vis, f"Seg error: {str(e)[:50]}", (10, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            seg_step += 1

            # FPS
            fps_counter += 1
            if time.time() - fps_time > 1.0:
                _fps = fps_counter
                fps_counter = 0
                fps_time = time.time()

            with _frame_lock:
                _latest_rgb = overlay
                _latest_depth = depth_color
                if seg_vis is not None:
                    _latest_seg = seg_vis
                _detection_info = det_text
                app.config['OBJ_INFO'] = obj_info_str

        except Exception as e:
            print(f"[VIEWER] Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.5)


def gen_stream(frame_type):
    """Generator for MJPEG stream."""
    import cv2
    while True:
        with _frame_lock:
            if frame_type == 'rgb':
                frame = _latest_rgb
            elif frame_type == 'depth':
                frame = _latest_depth
            elif frame_type == 'seg':
                frame = _latest_seg
            else:
                frame = None

        if frame is None:
            time.sleep(0.1)
            continue

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.05)


@app.route('/')
def index():
    return render_template_string(HTML_PAGE,
                                  port=app.config.get('PORT', 8080),
                                  segmentation=app.config.get('SEG', False))


@app.route('/stream/rgb')
def stream_rgb():
    return Response(gen_stream('rgb'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream/depth')
def stream_depth():
    return Response(gen_stream('depth'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream/seg')
def stream_seg():
    return Response(gen_stream('seg'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    return _json.dumps({
        'detection': _detection_info or 'No data',
        'objects': app.config.get('OBJ_INFO', '-'),
        'fps': _fps,
    }), 200, {'Content-Type': 'application/json'}


def main():
    parser = argparse.ArgumentParser(description='Rover Camera Viewer')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--segmentation', action='store_true',
                        help='Enable terrain segmentation mask view')
    parser.add_argument('--seg-model', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'mark_model', 'unet_lunar_segmentation.pth'))
    args = parser.parse_args()

    app.config['PORT'] = args.port
    app.config['SEG'] = args.segmentation

    t = threading.Thread(target=camera_thread,
                         args=(args.segmentation, args.seg_model),
                         daemon=True)
    t.start()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  Team Crater — Camera Viewer                             ║
║  Open in browser: http://0.0.0.0:{args.port:<5}                ║
║  Detection: DNN Face + Haar + MobileNet Person/Objects   ║
║  Segmentation: {'ON' if args.segmentation else 'OFF':42} ║
╚══════════════════════════════════════════════════════════╝
""")

    app.run(host='0.0.0.0', port=args.port, threaded=True)


if __name__ == '__main__':
    main()

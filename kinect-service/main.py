#!/usr/bin/env python3
"""Sage Kinect Service — skeleton, presence, zone detection with live dashboard."""

import asyncio
import ctypes
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import freenect
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker.task")

# --- Hardware (ctypes for tilt/LED — broken on K4W, kept for Xbox Kinect compat) ---
try:
    _sync = ctypes.CDLL("libfreenect_sync.so")
    _sync.freenect_sync_set_tilt_degs.argtypes = [ctypes.c_int, ctypes.c_int]
    _sync.freenect_sync_set_tilt_degs.restype = ctypes.c_int
    _sync.freenect_sync_set_led.argtypes = [ctypes.c_int, ctypes.c_int]
    _sync.freenect_sync_set_led.restype = ctypes.c_int
    HAS_MOTOR = True
except OSError:
    HAS_MOTOR = False

# --- MediaPipe Pose (Tasks API) ---
POSE_CONNS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
]
KEY_POINTS = {0, 11, 12, 15, 16, 23, 24, 27, 28}
NUM_LANDMARKS = 33


# --- Landmark smoothing (EMA) ---
class LandmarkSmoother:
    """Exponential moving average smoother for pose landmarks."""

    def __init__(self, alpha=0.4):
        self.alpha = alpha  # 0 = full smoothing, 1 = no smoothing
        self.prev = None

    def smooth(self, landmarks):
        """Smooth landmark positions. Returns list of dicts with smoothed x,y,z,v."""
        curr = [
            {"x": l.x, "y": l.y, "z": l.z, "v": l.visibility}
            for l in landmarks
        ]
        if self.prev is None:
            self.prev = curr
            return curr

        smoothed = []
        for i in range(len(curr)):
            if curr[i]["v"] < 0.3:
                smoothed.append(curr[i])
            else:
                smoothed.append({
                    "x": self.alpha * curr[i]["x"] + (1 - self.alpha) * self.prev[i]["x"],
                    "y": self.alpha * curr[i]["y"] + (1 - self.alpha) * self.prev[i]["y"],
                    "z": self.alpha * curr[i]["z"] + (1 - self.alpha) * self.prev[i]["z"],
                    "v": curr[i]["v"],
                })
        self.prev = smoothed
        return smoothed

    def reset(self):
        self.prev = None


# --- Depth temporal smoother ---
class DepthSmoother:
    """Rolling average over N depth frames to reduce noise."""

    def __init__(self, window=3):
        self.window = window
        self.frames = []

    def smooth(self, depth):
        self.frames.append(depth.astype(np.float32))
        if len(self.frames) > self.window:
            self.frames.pop(0)
        if len(self.frames) == 1:
            return depth
        stack = np.stack(self.frames, axis=0)
        return np.mean(stack, axis=0).astype(np.uint16)


# --- Constants ---
JPEG_Q = [cv2.IMWRITE_JPEG_QUALITY, 80]
COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO, "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA, "jet": cv2.COLORMAP_JET,
    "ocean": cv2.COLORMAP_OCEAN,
}


# --- Shared state ---
class State:
    def __init__(self):
        self._lock = threading.Lock()
        self.frame_id = 0
        self.rgb_jpeg: Optional[bytes] = None
        self.depth_jpeg: Optional[bytes] = None
        self.presence = False
        self.confidence = 0.0
        self.zone_h = "unknown"
        self.zone_d = "unknown"
        self.distance_mm = 0
        self.landmarks: Optional[list] = None
        self.depth_mode = "turbo"
        self.fps = 0.0

    def update(self, rgb_j, depth_j, presence, conf, zh, zd, dist, lm, fps):
        with self._lock:
            self.frame_id += 1
            self.rgb_jpeg = rgb_j
            self.depth_jpeg = depth_j
            self.presence = presence
            self.confidence = conf
            self.zone_h = zh
            self.zone_d = zd
            self.distance_mm = dist
            self.landmarks = lm
            self.fps = fps

    def snap(self):
        with self._lock:
            return dict(
                frame_id=self.frame_id,
                rgb_jpeg=self.rgb_jpeg, depth_jpeg=self.depth_jpeg,
                presence=self.presence, confidence=self.confidence,
                zone=dict(horizontal=self.zone_h, depth=self.zone_d,
                          distance_mm=self.distance_mm),
                landmarks=self.landmarks, fps=self.fps,
            )


state = State()


# --- Depth rendering ---
def render_depth(depth: np.ndarray) -> np.ndarray:
    mode = state.depth_mode
    no_data = depth >= 2047
    valid = depth[~no_data]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    vmin, vmax = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
    if vmax <= vmin:
        vmax = vmin + 1

    d = np.clip(depth.astype(np.float32), vmin, vmax)
    d = ((d - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    d[no_data] = 0

    if mode in COLORMAPS:
        out = cv2.applyColorMap(d, COLORMAPS[mode])
    elif mode == "gray":
        out = cv2.cvtColor(255 - d, cv2.COLOR_GRAY2BGR)
    elif mode == "edges":
        edges = cv2.Canny(cv2.GaussianBlur(d, (5, 5), 0), 40, 120)
        out = np.zeros((*d.shape, 3), dtype=np.uint8)
        out[:, :, 1] = edges
    else:
        out = cv2.applyColorMap(d, cv2.COLORMAP_TURBO)

    out[no_data] = [0, 0, 0]
    return out


# --- Zone detection ---
def raw_to_mm(raw: float) -> int:
    if raw >= 2047 or raw <= 0:
        return 0
    try:
        return int(1000.0 / (-0.00307 * raw + 3.33))
    except (ZeroDivisionError, ValueError):
        return 0


def compute_zone(landmarks, depth, w=640, h=480):
    """Compute zone from smoothed landmark dicts."""
    lh, rh = landmarks[23], landmarks[24]
    cx = int((lh["x"] + rh["x"]) / 2 * w)
    cy = int((lh["y"] + rh["y"]) / 2 * h)

    zh = "left" if cx < w / 3 else ("center" if cx < 2 * w / 3 else "right")

    y1, y2 = max(0, cy - 15), min(h, cy + 15)
    x1, x2 = max(0, cx - 15), min(w, cx + 15)
    region = depth[y1:y2, x1:x2]
    valid = region[(region > 0) & (region < 2047)]
    dist = raw_to_mm(float(np.median(valid))) if valid.size > 0 else 0

    if dist <= 0:
        zd = "unknown"
    elif dist < 1500:
        zd = "near"
    elif dist < 3000:
        zd = "mid"
    else:
        zd = "far"

    return zh, zd, dist


# --- Skeleton drawing (overlaid on RGB, uses smoothed dicts) ---
def draw_skeleton(img, landmarks, w=640, h=480):
    """Draw skeleton from smoothed landmark dicts (keys: x, y, z, v)."""
    out = img.copy()

    for start, end in POSE_CONNS:
        s, e = landmarks[start], landmarks[end]
        if s["v"] < 0.3 or e["v"] < 0.3:
            continue
        pt1 = (int(s["x"] * w), int(s["y"] * h))
        pt2 = (int(e["x"] * w), int(e["y"] * h))
        cv2.line(out, pt1, pt2, (0, 255, 136), 2, cv2.LINE_AA)

    for i, lm in enumerate(landmarks):
        if lm["v"] < 0.3:
            continue
        pt = (int(lm["x"] * w), int(lm["y"] * h))
        r = 5 if i in KEY_POINTS else 3
        cv2.circle(out, pt, r, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(out, pt, r, (0, 100, 200), 1, cv2.LINE_AA)

    return out


# --- Capture thread ---
def capture_loop():
    frame_times = []
    lm_smoother = LandmarkSmoother(alpha=0.4)  # 0.4 = moderate smoothing
    # Depth smoothing removed — causes ghosting/doubling on movement

    landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    start_time = time.monotonic()
    print("Capture thread ready — heavy model loaded", flush=True)

    while True:
        try:
            rgb, _ = freenect.sync_get_video()
            depth_raw, _ = freenect.sync_get_depth()
            if rgb is None or depth_raw is None:
                time.sleep(0.05)
                continue

            depth = depth_raw
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # MediaPipe detection
            ts_ms = int((time.monotonic() - start_time) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            presence = len(result.pose_landmarks) > 0
            conf, zh, zd, dist, lm_list = 0.0, "unknown", "unknown", 0, None

            display_bgr = rgb_bgr
            if presence:
                raw_lm = result.pose_landmarks[0]
                conf = round(float(np.mean([l.visibility for l in raw_lm])), 3)

                # Smooth landmarks for stable display
                smoothed = lm_smoother.smooth(raw_lm)
                zh, zd, dist = compute_zone(smoothed, depth)
                lm_list = [
                    {"x": round(l["x"], 4), "y": round(l["y"], 4),
                     "z": round(l["z"], 4), "v": round(l["v"], 3)}
                    for l in smoothed
                ]
                display_bgr = draw_skeleton(rgb_bgr, smoothed)
                cv2.putText(display_bgr, f"{zh} / {zd}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)
                if dist > 0:
                    cv2.putText(display_bgr, f"{dist / 1000:.1f}m", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)
            else:
                lm_smoother.reset()

            # Encode
            _, rb = cv2.imencode(".jpg", display_bgr, JPEG_Q)
            _, db = cv2.imencode(".jpg", render_depth(depth), JPEG_Q)

            # FPS
            now = time.monotonic()
            frame_times = [t for t in frame_times if t > now - 1.0]
            frame_times.append(now)

            state.update(rb.tobytes(), db.tobytes(),
                         presence, conf, zh, zd, dist, lm_list, len(frame_times))

        except Exception as e:
            print(f"capture error: {e}", flush=True)
            time.sleep(0.5)


# --- MJPEG streaming ---
async def mjpeg_gen(feed: str):
    last_id = 0
    while True:
        s = state.snap()
        if s["frame_id"] > last_id:
            last_id = s["frame_id"]
            jpeg = s.get(f"{feed}_jpeg")
            if jpeg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode()
                       + b"\r\n\r\n" + jpeg + b"\r\n")
        await asyncio.sleep(0.05)


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    print("Sage Kinect Service started", flush=True)
    yield


# --- App ---
app = FastAPI(title="Sage Kinect Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# --- API ---
@app.get("/feed/{stream}")
async def feed(stream: str):
    if stream not in ("rgb", "depth"):
        return JSONResponse({"error": "unknown stream, use rgb or depth"}, 404)
    return StreamingResponse(mjpeg_gen(stream),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/status")
async def get_status():
    s = state.snap()
    return {
        "ok": True, "fps": s["fps"],
        "presence": s["presence"], "confidence": s["confidence"],
        "zone": s["zone"], "depth_mode": state.depth_mode,
    }


@app.get("/detections")
async def get_detections():
    s = state.snap()
    return {
        "frame_id": s["frame_id"],
        "presence": s["presence"], "confidence": s["confidence"],
        "zone": s["zone"], "landmarks": s["landmarks"],
    }


class TiltReq(BaseModel):
    angle: int


@app.post("/tilt")
async def post_tilt(req: TiltReq):
    if not HAS_MOTOR:
        return {"ok": False, "error": "motor not available"}
    a = max(-27, min(27, req.angle))
    ret = _sync.freenect_sync_set_tilt_degs(a, 0)
    return {"ok": ret == 0, "angle": a}


class LedReq(BaseModel):
    led: int


@app.post("/led")
async def post_led(req: LedReq):
    if not HAS_MOTOR:
        return {"ok": False, "error": "LED not available"}
    ret = _sync.freenect_sync_set_led(req.led, 0)
    return {"ok": ret == 0, "led": req.led}


class VizReq(BaseModel):
    mode: str


@app.post("/viz")
async def post_viz(req: VizReq):
    state.depth_mode = req.mode
    return {"ok": True, "mode": req.mode}


# --- WebSocket ---
@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    await ws.accept()
    last_id = 0
    try:
        while True:
            s = state.snap()
            if s["frame_id"] > last_id:
                last_id = s["frame_id"]
                await ws.send_json({
                    "frame_id": s["frame_id"],
                    "presence": s["presence"],
                    "confidence": s["confidence"],
                    "zone": s["zone"],
                    "landmarks": s["landmarks"],
                    "fps": s["fps"],
                })
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass


# --- Dashboard ---
PAGE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sage</title>
<style>
:root{--bg:#0a0a0a;--surface:#111;--border:#1a1a1a;--accent:#00ff88;--accent2:#00c8ff;--dim:#555;--text:#ccc}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'SF Mono','Fira Code','Cascadia Code',monospace;overflow:hidden;height:100vh}
.hdr{padding:8px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:16px;height:40px}
.hdr h1{font-size:16px;font-weight:400;letter-spacing:3px}
.hdr h1 b{color:var(--accent);font-weight:600}
.hdr .info{margin-left:auto;font-size:11px;color:var(--dim);display:flex;gap:16px}
.hdr .info span b{color:var(--accent)}
.main{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr auto;height:calc(100vh - 40px)}
.cell{position:relative;background:#050505;overflow:hidden}
.cell img{width:100%;height:100%;object-fit:contain}
.tag{position:absolute;top:8px;left:8px;background:rgba(0,0,0,.85);padding:3px 10px;font-size:10px;
  letter-spacing:1.5px;border-left:2px solid var(--accent);z-index:2;color:var(--text)}
.bar{grid-column:1/-1;background:var(--surface);border-top:1px solid var(--border);
  display:flex;align-items:center;padding:6px 16px;gap:16px;flex-wrap:wrap;font-size:11px}
.status-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:4px}
.status-dot.on{background:var(--accent);box-shadow:0 0 6px var(--accent)}
.status-dot.off{background:#333}
.zone-tag{background:#1a1a1a;border:1px solid #333;padding:2px 8px;border-radius:3px;
  color:var(--accent2);letter-spacing:1px}
.sep{width:1px;height:16px;background:#222}
.controls{display:flex;align-items:center;gap:6px;margin-left:auto}
.btn{background:#1a1a1a;border:1px solid #333;color:var(--text);padding:3px 8px;font-size:10px;
  letter-spacing:0.5px;cursor:pointer;border-radius:3px;font-family:inherit}
.btn:hover{background:#222;border-color:var(--accent)}
.btn.active{border-color:var(--accent);color:var(--accent)}
</style>
</head><body>
<div class="hdr">
  <h1><b>SAGE</b> KINECT</h1>
  <div class="info">
    <span>FPS <b id="fps">0</b></span>
    <span>CONF <b id="conf">0</b></span>
  </div>
</div>
<div class="main">
  <div class="cell">
    <div class="tag">RGB + SKELETON</div>
    <img src="/feed/rgb">
  </div>
  <div class="cell">
    <div class="tag" id="dtag">DEPTH // TURBO</div>
    <img src="/feed/depth">
  </div>
  <div class="bar">
    <span class="status-dot off" id="pDot"></span>
    <span id="pText">No person</span>
    <div class="sep"></div>
    <span class="zone-tag" id="zoneTag">&mdash;</span>
    <span id="distText"></span>
    <div class="sep"></div>
    <div class="controls" id="vizBtns">
      <span style="color:var(--dim)">DEPTH</span>
      <button class="btn active" onclick="setViz('turbo',this)">TURBO</button>
      <button class="btn" onclick="setViz('inferno',this)">INFERNO</button>
      <button class="btn" onclick="setViz('plasma',this)">PLASMA</button>
      <button class="btn" onclick="setViz('gray',this)">GRAY</button>
      <button class="btn" onclick="setViz('edges',this)">EDGES</button>
    </div>
  </div>
</div>
<script>
const fpsEl = document.getElementById('fps');
const confEl = document.getElementById('conf');
const pDot = document.getElementById('pDot');
const pText = document.getElementById('pText');
const zoneTag = document.getElementById('zoneTag');
const distText = document.getElementById('distText');

// WebSocket for status updates
function connectWs() {
  const ws = new WebSocket(`ws://${location.host}/ws/events`);
  ws.onmessage = (e) => {
    const d = JSON.parse(e.data);
    fpsEl.textContent = d.fps;
    confEl.textContent = (d.confidence * 100).toFixed(0) + '%';
    if (d.presence) {
      pDot.className = 'status-dot on';
      pText.textContent = 'Person detected';
      zoneTag.textContent = `${d.zone.horizontal.toUpperCase()} / ${d.zone.depth.toUpperCase()}`;
      distText.textContent = d.zone.distance_mm > 0
        ? `${(d.zone.distance_mm / 1000).toFixed(1)}m` : '';
    } else {
      pDot.className = 'status-dot off';
      pText.textContent = 'No person';
      zoneTag.textContent = '\u2014';
      distText.textContent = '';
    }
  };
  ws.onclose = () => setTimeout(connectWs, 2000);
  ws.onerror = () => ws.close();
}
connectWs();

function setViz(mode, el) {
  document.querySelectorAll('#vizBtns .btn').forEach(b => b.classList.remove('active'));
  if (el) el.classList.add('active');
  document.getElementById('dtag').textContent = 'DEPTH // ' + mode.toUpperCase();
  fetch('/viz', {method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mode:mode})});
}
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

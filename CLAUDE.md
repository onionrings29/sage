# Sage — Activity Monitoring System

Kinect v1 depth + RGB for presence detection, skeleton tracking, zone detection, and (future) posture/behavioral analysis. Runs on Proxmox homelab.

## Architecture

Single Python service (FastAPI + MediaPipe + libfreenect) on CT 144. Future: separate dashboard (Next.js) and brain layer (Sage) that classifies and acts on detector data.

```
Kinect v1 (USB) → libfreenect → MediaPipe Pose → FastAPI
                                                   ├── MJPEG feeds (RGB+skeleton, depth)
                                                   ├── WebSocket (real-time detections)
                                                   └── REST API (status, config)
```

## Hardware

- **Kinect for Windows v1** (model 1517) on homelab01 USB
  - Motor PID 045e:02c2 — USB hub class, tilt/LED NOT supported (K4W limitation, only works on Xbox Kinect 360 model 1414)
  - Camera PID 045e:02ae, Audio PID 045e:02ad
- **CT 144**: Debian 13, 10 cores, 16GB RAM, iGPU passthrough, USB bus passthrough

## Server Access

```bash
# homelab01 (Proxmox host)
ssh root@10.129.20.12

# CT 144 (Kinect service container)
ssh root@10.129.20.12 'pct exec 144 -- bash'
# Or direct SSH:
sshpass -p 'awantasayo' ssh root@10.129.20.88

# Service management
ssh root@10.129.20.12 'pct exec 144 -- systemctl restart sage-kinect'
ssh root@10.129.20.12 'pct exec 144 -- journalctl -u sage-kinect -f'
```

## Deploy

From this repo root:
```bash
./deploy.sh
```

This rsyncs `kinect-service/` to CT 144 at `/opt/sage/kinect-service/`, downloads the MediaPipe model if missing, and restarts the service.

**Service URL**: http://10.129.20.88:8080

## Development

- All code lives in this repo, deployed via rsync to CT 144
- The MediaPipe model file (`pose_landmarker.task`, 9MB) is gitignored — downloaded on first deploy
- Python dependencies are installed directly on CT 144 (no venv — it's a dedicated container)

### Key Files

| File | Purpose |
|------|---------|
| `kinect-service/main.py` | FastAPI service — capture, detection, streaming, dashboard |
| `kinect-service/requirements.txt` | Python dependencies |
| `kinect-service/models/` | MediaPipe model files (gitignored) |
| `deploy.sh` | Deploy script — rsync + restart |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard (embedded HTML) |
| `/feed/rgb` | GET | MJPEG — RGB with skeleton overlay |
| `/feed/depth` | GET | MJPEG — colorized depth |
| `/ws/events` | WS | Real-time detections (presence, zone, landmarks) |
| `/status` | GET | Health check + current state |
| `/detections` | GET | Full detection data + landmarks |
| `/tilt` | POST | Motor tilt (broken on K4W hardware) |
| `/led` | POST | LED control (broken on K4W hardware) |
| `/viz` | POST | Depth colormap mode |

### Detection Data (WebSocket)

```json
{
  "frame_id": 123,
  "presence": true,
  "confidence": 0.72,
  "zone": {"horizontal": "center", "depth": "near", "distance_mm": 1216},
  "landmarks": [{"x": 0.5, "y": 0.3, "z": 0.1, "v": 0.98}, ...],
  "fps": 15
}
```

## Known Issues

- **Tilt/LED broken**: K4W motor device (02c2) is a USB hub, not a vendor device. Standard libfreenect tilt/LED commands fail. Xbox Kinect 360 (model 1414) would work.
- **gspca_kinect kernel module**: Must be blacklisted on homelab01 (`/etc/modprobe.d/kinect.conf`) or it claims the camera device
- **Near mode**: K4W supports near mode (400mm min depth vs 800mm) but libfreenect Python wrapper doesn't expose `freenect_set_flag`. Not currently needed.

## Phases

| Phase | Scope |
|-------|-------|
| 1 (current) | Kinect service: skeleton, presence, zone detection, dashboard |
| 2 | Activity classification, sedentary alerts, PostgreSQL |
| 3 | ADHD/fidgeting metrics, study quality, posture scoring |
| 4 | Frigate camera integration in Sage dashboard |
| 5 | Cross-sensor behavioral synthesis (Sage brain) |

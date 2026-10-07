# CrowdPulse Backend

Detection and risk service for CrowdPulse AI. It reads a live video feed from a CCTV camera, a phone camera or a webcam, counts the people in view, works out the safe and maximum capacity of the monitored floor, and returns a crowd risk level.

The dashboard lives in the companion repository, `crowdpulse-frontend`.

## What it does

- Counts people in the live frame with YOLOv8.
- Detects the walkable floor with a segmentation model and converts it to an area in square metres.
- Derives safe and maximum capacity from crowd safety densities (1.0 and 1.5 people per m2).
- Scores risk from occupancy, growth rate and how long the zone stays crowded.
- Recalibrates the floor estimate automatically while people are in view.
- Connects to RTSP CCTV cameras and phone camera apps, and can switch camera at runtime.
- Logs every reading to SQLite for history charts.

## Requirements

- Python 3.10 or newer
- A camera: RTSP CCTV or IP camera, a phone running a camera app, or a webcam
- Internet access on the first start, to download the floor detection model once

## Quick start

```bash
git clone https://github.com/Charan7sai/crowdpulse-backend.git
cd crowdpulse-backend

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
pip install transformers

cp my.env.example my.env      # Windows: copy my.env.example my.env
# edit my.env and set CAMERA_SOURCE

python app.py
```

The service listens on `http://127.0.0.1:5001`.

The first start downloads the floor model (about 340 MB). Run it once with internet access before a demo or an offline deployment. After that it works offline.

## Connecting a camera

Set `CAMERA_SOURCE` in `my.env`, or choose the camera from the dashboard setup screen. A camera chosen on the dashboard is saved in `camera_settings.json` and takes priority after a restart.

| Camera | `CAMERA_SOURCE` example |
|---|---|
| Laptop webcam | `0` |
| Android phone (IP Webcam app) | `http://192.168.1.50:8080/video` |
| Phone or PC webcam app (DroidCam) | `http://192.168.1.50:4747/video` |
| Hikvision / HiWatch | `rtsp://user:password@192.168.1.64:554/Streaming/Channels/101` |
| Dahua / CP Plus / Amcrest / Lorex | `rtsp://user:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0` |

Built-in stream address patterns cover Hikvision, Dahua and its OEM brands, Axis, Uniview, Reolink, TP-Link Tapo, Foscam, Hanwha Wisenet, Vivotek, EZVIZ and generic RTSP cameras. `POST /camera/discover` tries every pattern against a camera and reports which ones open.

Tips:

- The computer running the backend must be able to reach the camera. On networks that isolate devices (guest, college or hotel Wi-Fi), use the phone's hotspot instead.
- Use the sub stream of a CCTV camera if processing lags.
- Large streams are scaled down to `CAMERA_MAX_WIDTH` before processing.
- Use a viewer account on the camera, not the admin account.
- Some cameras lock an account after several failed logins, so confirm the credentials before running discovery.

## Configuration

All settings are read from `my.env`.

| Variable | Default | Description |
|---|---|---|
| `CAMERA_SOURCE` | `0` | Webcam index, RTSP URL or HTTP/MJPEG URL |
| `CAMERA_MAX_WIDTH` | `960` | Streams wider than this are scaled down |
| `CAMERA_ADMIN_TOKEN` | empty | If set, camera and floor area changes require the header `X-Admin-Token` |
| `AUTO_CALIBRATE_ENABLED` | `true` | Turn automatic recalibration on or off |
| `AUTO_CALIBRATE_INTERVAL_SEC` | `30` | How often the calibrator checks the scene |
| `AUTO_CALIBRATE_SKIP_WHEN_EMPTY` | `true` | Skip recalibration when nobody is in view |
| `CALIBRATION_SMOOTHING` | `0.6` | Blend of previous and new area estimates (0 disables) |
| `FLOOR_MODEL_ID` | `nvidia/segformer-b5-finetuned-ade-640-640` | Floor segmentation model. `none` uses YOLOv8-seg object subtraction instead |
| `FLOOR_REFRESH_SEC` | `300` | How often the floor model re-runs. Manual recalibration and camera changes always re-run it |
| `ZONE_AREA_OVERRIDE_M2` | empty | Real floor area in m2. Overrides the camera estimate |
| `ZONE_VIEW_HEIGHT_M` | `5.0` | Assumed height of the camera view, used only when no real area is set |
| `ZONE_NAME` | `Main Hall` | Name shown on the dashboard |
| `ZONE_AREA_M2` | `50` | Fallback area if calibration fails |
| `YOLO_MODEL_PATH` | `yolov8n.pt` | Person detection model |
| `YOLO_SEG_MODEL_PATH` | `yolov8n-seg.pt` | Segmentation model for person footprints and the fallback floor method |
| `YOLO_CONFIDENCE` | `0.4` | Detection confidence threshold |
| `FLASK_PORT` | `5001` | Port the service listens on |
| `DATABASE_PATH` | `crowd_data.db` | SQLite file for readings |

A larger person model (`yolov8s.pt` or `yolov8m.pt`) improves counts at the cost of speed on CPU.

## How capacity is calculated

1. The floor model labels each pixel of the frame. Pixels labelled floor, rug or carpet form the walkable area. People on the floor count as floor, so a crowd does not shrink the area.
2. The pixel area is converted to square metres. If a real area is entered (in `my.env` or on the dashboard) it is used directly. Otherwise the area is estimated from the camera view, which is approximate and works best for cameras mounted high and looking down.
3. Safe capacity is the area multiplied by 1.0 person per m2. Maximum capacity is the area multiplied by 1.5 people per m2.

For reliable numbers, enter the real floor area.

## How risk is scored

Each reading is evaluated by the risk engine:

- **Occupancy** is the smoothed count (exponential moving average, alpha 0.7) divided by capacity.
- **Growth** is the change in count compared with 3 readings earlier. A gain of 3 or more people raises the surge flag.
- **Persistence** is how long occupancy stays above 0.8. It exits that state when occupancy falls below 0.7.
- **Risk score** is `0.5 x occupancy + 0.3 x growth + 0.2 x persistence`.

| Score | Level |
|---|---|
| below 0.4 | Safe |
| 0.4 to 0.6 | Elevated |
| 0.6 to 0.8 | High |
| 0.8 and above | Critical |

Density in people per m2 is also classified with Fruin levels of service: free movement (below 0.5), restricted (below 1.0), body contact possible (below 2.0), pushing and pressure (below 4.0) and crush risk (4.0 and above).

## Automatic recalibration

Every `AUTO_CALIBRATE_INTERVAL_SEC` seconds the calibrator checks the scene:

- If the zone has never been calibrated, it calibrates as soon as the camera is ready.
- If people are in view, it recalibrates, because furniture and crowds move.
- If the floor is empty and already calibrated, it skips the run.
- After a camera change it calibrates immediately.

The manual `POST /recalibrate` call shares the same code path, so manual and automatic runs never overlap.

## API

All responses are JSON unless noted.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/detect` | Detect people in the latest frame and return the full risk assessment |
| GET | `/history?minutes=N` | Logged readings for the last N minutes (1 to 60) |
| GET | `/video_feed` | MJPEG stream with boxes around detected people |
| GET | `/zone` | Floor area, safe and maximum capacity, calibration details |
| GET | `/zone_preview` | MJPEG stream with the detected floor highlighted |
| POST | `/zone/area` | Set the real floor area: `{"area_m2": 12}`. Send `null` to use the estimate |
| POST | `/recalibrate` | Recalibrate the zone on the current frame |
| GET | `/calibration_status` | Automatic recalibration state |
| GET | `/camera/status` | Camera health and the current source (password hidden) |
| GET | `/camera/presets` | Supported camera brands and phone apps |
| POST | `/camera/test` | Test a camera without switching to it |
| POST | `/camera/source` | Switch camera using a `url`, or `brand`, `ip`, `user`, `password`, `port`, `channel`, `stream` |
| POST | `/camera/discover` | Try every known stream address for one camera |

`/detect` returns the current and smoothed count, occupancy ratio, growth rate, risk score, risk level, surge flag, time spent in the high state and a timestamp.

Example:

```bash
curl http://127.0.0.1:5001/detect

curl -X POST http://127.0.0.1:5001/camera/source \
  -H "Content-Type: application/json" \
  -d '{"brand":"hikvision","ip":"192.168.1.64","user":"viewer","password":"secret","stream":"main"}'
```

## Security notes

- `my.env` and `camera_settings.json` can contain camera passwords. Both are excluded from Git. Never commit them.
- Set `CAMERA_ADMIN_TOKEN` if the service is reachable by anyone other than you. Without it, anyone who can reach the port can change the camera.
- The Flask development server is used by default. For a permanent deployment, run behind a production WSGI server and a reverse proxy with HTTPS.

## Project structure

```
app.py               Flask app and endpoints
config.py            Settings and thresholds
camera_source.py     Camera reader with reconnect and runtime switching
camera_presets.py    Stream address patterns and discovery
zone_estimator.py    Floor detection, area and capacity
auto_calibration.py  Automatic recalibration scheduler
risk_engine.py       Stateful risk scoring
database.py          SQLite logging
my.env.example       Example configuration
```

## Troubleshooting

| Problem | What to check |
|---|---|
| `Cannot open camera source` | Wrong address or credentials, or the camera is unreachable from this computer |
| Phone camera does not connect | Open the phone's address in the computer's browser first. If that fails, switch to the phone's hotspot |
| Floor model errors on start | Run `pip install transformers` and make sure the first start has internet. Set `FLOOR_MODEL_ID=none` to run without it |
| Capacity looks too high | Enter the real floor area on the dashboard or set `ZONE_AREA_OVERRIDE_M2` |
| Video lags | Use the camera's sub stream, lower `CAMERA_MAX_WIDTH`, or keep `yolov8n.pt` |

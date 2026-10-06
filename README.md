# AI Detection Service — Crowd Management Backend

A real-time crowd monitoring and risk assessment service powered by YOLOv8 and Flask. This backend service provides intelligent crowd density analysis, risk scoring, and live video streaming capabilities for crowd management applications.

## 🎯 Overview

The AI Detection Service uses computer vision and machine learning to:
- **Detect and count people** in real-time using YOLOv8
- **Assess crowd risk** based on density, growth rate, and persistence metrics
- **Stream live video** with MJPEG encoding
- **Log historical data** to SQLite for trend analysis
- **Provide REST API endpoints** for frontend integration

## ✨ Features

### Real-Time Detection
- YOLOv8-based person detection with configurable confidence threshold
- Continuous camera capture on background thread
- Frame-level inference with minimal latency

### Intelligent Risk Assessment
The risk engine evaluates three key factors:

1. **Density Ratio** — Current occupancy vs. zone capacity (smoothed with EMA)
2. **Growth Rate** — Rate of crowd increase over sliding time window
3. **Persistence** — Duration of high-density state with hysteresis

Risk levels: `Safe` → `Elevated` → `High` → `Critical`

### Data Persistence
- SQLite database for logging all crowd readings
- Historical data retrieval for trend charts
- Thread-safe database operations

### Video Streaming
- MJPEG stream endpoint for live camera feed
- ~20 FPS streaming with JPEG encoding
- Compatible with standard HTML `<img>` tags

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Flask REST API Server           │
├─────────────────────────────────────────┤
│  GET /detect      → Risk Assessment     │
│  GET /history     → Historical Data     │
│  GET /video_feed  → MJPEG Stream        │
└─────────────────────────────────────────┘
           ↓                ↓
    ┌──────────┐    ┌──────────────┐
    │  YOLOv8  │    │ Risk Engine  │
    │  Model   │    │  (Stateful)  │
    └──────────┘    └──────────────┘
           ↓                ↓
    ┌──────────────────────────┐
    │   Camera Capture Thread  │
    │   (Background, Daemon)   │
    └──────────────────────────┘
                     ↓
              ┌──────────────┐
              │ SQLite DB    │
              │ crowd_data   │
              └──────────────┘
```

## 📋 Prerequisites

- **Python 3.8+**
- **Webcam or IP Camera** (accessible via OpenCV)
- **OS**: Windows, Linux, or macOS

## 🚀 Installation

### 1. Clone the Repository
```bash
cd ai-service
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

The YOLOv8n model (`yolov8n.pt`) will be automatically downloaded on first run if not present.

## ⚙️ Configuration

Edit `config.py` to customize the service:

### Zone Settings
```python
ZONE_CAPACITY = 5  # Maximum safe occupancy for monitored area
```

### Camera Settings
```python
CAMERA_INDEX = 1            # 0 = default webcam, 1 = external camera
YOLO_CONFIDENCE_THRESHOLD = 0.4  # Min confidence for detections
```

### Risk Engine Parameters
```python
# EMA Smoothing
SMOOTHING_ALPHA = 0.7  # Weight for current count (higher = more reactive)

# Growth Rate
GROWTH_RATE_WINDOW = 3  # Compare against count N readings ago

# Persistence & Hysteresis
HIGH_DENSITY_THRESHOLD = 0.8   # Enter high-density state
HIGH_DENSITY_EXIT = 0.7        # Exit high-density state (prevents flickering)
CRITICAL_PERSISTENCE_SEC = 10  # Seconds in high state → Critical

# Surge Detection
SURGE_GROWTH_THRESHOLD = 3  # Person increase triggering surge flag

# Risk Score Weights (must sum to 1.0)
WEIGHT_DENSITY = 0.5
WEIGHT_GROWTH = 0.3
WEIGHT_PERSISTENCE = 0.2

# Risk Level Thresholds
RISK_THRESHOLD_LOW = 0.4     # Score < 0.4 → Safe
RISK_THRESHOLD_MEDIUM = 0.6  # Score < 0.6 → Elevated
RISK_THRESHOLD_HIGH = 0.8    # Score < 0.8 → High
                              # Score ≥ 0.8 → Critical
```

### Server Settings
```python
FLASK_PORT = 5001
DATABASE_PATH = "crowd_data.db"
```

## 🔌 API Endpoints

### `GET /detect`
Run YOLO inference on the latest camera frame and return risk assessment.

**Response:**
```json
{
  "current_count": 8,
  "smoothed_count": 7.3,
  "density_ratio": 0.73,
  "growth_rate": 2.0,
  "risk_score": 0.65,
  "risk_level": "Elevated",
  "surge_flag": false,
  "duration_in_high_state": 0.0,
  "timestamp": "2026-02-17T14:32:18.123456+00:00"
}
```

**Status Codes:**
- `200` — Success
- `503` — Camera not ready

---

### `GET /history?minutes=N`
Retrieve recent crowd log entries for trend analysis.

**Query Parameters:**
- `minutes` (optional) — Time range in minutes (default: 2, range: 1-60)

**Response:**
```json
[
  {
    "id": 1,
    "timestamp": "2026-02-17T14:30:00.000000+00:00",
    "current_count": 5,
    "density_ratio": 0.5,
    "risk_level": "Safe",
    "risk_score": 0.35,
    ...
  },
  ...
]
```

---

### `GET /video_feed`
MJPEG stream of live camera feed (~20 FPS).

**Response:**
- Content-Type: `multipart/x-mixed-replace; boundary=frame`

**Usage in HTML:**
```html
<img src="http://localhost:5001/video_feed" alt="Live Feed" />
```

## 🎮 Usage

### Start the Service
```bash
python app.py
```

Expected output:
```
[INFO] Camera 1 opened — capture thread running
 * Running on http://127.0.0.1:5001
```

### Test the API
```bash
# Get current risk assessment
curl http://localhost:5001/detect

# Get last 5 minutes of history
curl http://localhost:5001/history?minutes=5

# Stream video (open in browser)
# http://localhost:5001/video_feed
```

## 🧠 Risk Engine Details

### Risk Score Calculation

The compound risk score is a weighted sum:

$$
\text{risk\_score} = w_d \cdot r_d + w_g \cdot r_g + w_p \cdot r_p
$$

Where:
- $r_d$ = **Density Risk** = `min(density_ratio, 1.0)`
- $r_g$ = **Growth Risk** = `min(growth_rate / 10, 1.0)`
- $r_p$ = **Persistence Risk** = `min(duration / 30, 1.0)`

Default weights: $w_d = 0.5$, $w_g = 0.3$, $w_p = 0.2$

### Risk Level Classification

| Risk Level  | Score Range | Description                          |
|-------------|-------------|--------------------------------------|
| **Safe**    | < 0.4       | Normal operation, no concerns        |
| **Elevated**| 0.4 - 0.6   | Increased density, monitor closely   |
| **High**    | 0.6 - 0.8   | High density, prepare intervention   |
| **Critical**| ≥ 0.8       | Overcrowded, immediate action needed |

### Surge Detection

A surge is flagged when:
```python
growth_rate >= SURGE_GROWTH_THRESHOLD
```

This indicates a rapid influx of people requiring immediate attention.

### Hysteresis

To prevent oscillation between risk states, the engine uses hysteresis:
- **Enter high state** when `density_ratio >= 0.8`
- **Exit high state** only when `density_ratio < 0.7`

This creates a "sticky" behavior that reduces false alarms.

## 📁 Project Structure

```
ai-service/
├── app.py              # Flask server & API endpoints
├── config.py           # Configuration parameters
├── database.py         # SQLite persistence layer
├── risk_engine.py      # Stateful risk assessment engine
├── requirements.txt    # Python dependencies
├── yolov8n.pt         # YOLOv8 nano model weights
├── crowd_data.db      # SQLite database (auto-generated)
└── README.md          # This file
```

## 🔧 Troubleshooting

### Camera Not Found
```
[ERROR] Cannot open camera index 1
```
**Solution:** Change `CAMERA_INDEX` in `config.py` to `0` (default webcam).

### Low Detection Accuracy
**Solution:** Adjust `YOLO_CONFIDENCE_THRESHOLD` in `config.py`. Lower values detect more (but less confident) objects.

### Port Already in Use
```
Address already in use
```
**Solution:** Change `FLASK_PORT` in `config.py` or kill the process using port 5001.

### Database Locked
**Solution:** Ensure only one instance of the service is running. SQLite doesn't support concurrent writers well.

## 🔒 Security Notes

⚠️ **This is a development server.** For production:
- Use a production WSGI server (gunicorn, uWSGI)
- Add authentication/authorization
- Enable HTTPS/TLS
- Restrict CORS origins
- Rate-limit API endpoints

## 📊 Performance

- **Detection Speed:** ~30-50ms per frame (YOLO inference)
- **Video Stream:** ~20 FPS with JPEG encoding
- **API Response Time:** < 100ms (excluding inference)
- **Memory Usage:** ~500MB (YOLOv8n model loaded)

## 🤝 Integration

This backend is designed to work with the Next.js frontend in `../frontend-next`. The frontend consumes:
- `/detect` endpoint for real-time risk display
- `/history` endpoint for trend charts
- `/video_feed` endpoint for live camera view

## 📝 License

[Add your license here]

## 👥 Contributing

[Add contribution guidelines here]

---

**Built with:** Python • Flask • YOLOv8 • OpenCV • SQLite

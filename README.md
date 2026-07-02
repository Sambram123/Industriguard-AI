# 🏭 IndustriGuard AI

**AI-powered industrial safety surveillance system** that verifies employee identity via QR codes and enforces full PPE (Personal Protective Equipment) compliance using real-time computer vision. Results are reported to a Flask/SQLite backend and visualized on a live React dashboard.

---

## ✨ Features

- **QR-based employee identification** — Workers scan their QR ID cards to start the safety check.
- **Full PPE detection** — YOLOv8 model detects **5 PPE items** per worker:
  | # | PPE Item | Detection Zone |
  |---|----------|----------------|
  | 1 | Helmet | Expanded upward from person bbox |
  | 2 | Safety Vest | Within person bbox |
  | 3 | Gloves | Within person bbox |
  | 4 | Goggles | Expanded upward from person bbox |
  | 5 | Boots | Expanded downward from person bbox |
- **Per-person compliance** — Each detected person gets an individual safety score (`0–100%`) based on how many of the 5 items are present.
- **Multi-person tracking** — ByteTrack integration allows simultaneous monitoring of multiple workers in the frame.
- **Real-time dashboard** — React SPA with WebSocket updates, live alerts, trend charts, department analytics, and Excel export.
- **Dual reporting** — Results are saved to both a local Excel workbook and the backend database.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AI Station (ai/)                        │
│                                                             │
│  Camera Feed ──► QR Scanner ──► PPE Detector (YOLOv8)       │
│                                      │                      │
│                              Safety Status Engine           │
│                              (5-item compliance)            │
│                                 │          │                │
│                          Excel Report   HTTP POST           │
│                                            │                │
└────────────────────────────────────────────┼────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend (backend/)                        │
│                                                             │
│  Flask + Socket.IO ──► SQLite (logs + latest status)        │
│       │                                                     │
│       ├── REST API (stats, trend, departments, checks)      │
│       └── WebSocket broadcast (check_update event)          │
│                                                             │
└────────────────────────────────────────────┬────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Frontend (frontend/)                       │
│                                                             │
│  React/Vite SPA                                             │
│       ├── Stat Cards (checks today, ready %, violations)    │
│       ├── Employee Table (per-item ✓/✗ + safety %)          │
│       ├── Trend Chart (24h hourly breakdown)                │
│       ├── Department Chart (compliance by dept)             │
│       ├── Check History (recent entries with safety %)      │
│       ├── Live Alert Popup (real-time toast notification)   │
│       └── Excel Export (client-side .xlsx download)         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🗂 Project Structure

```
Industriguard-AI/
├── ai/                          # AI safety station
│   ├── main_ai.py               # Main entry point — state machine
│   ├── ppe_detector.py          # YOLOv8 detection + per-person compliance
│   ├── safety_status.py         # Rule engine (5-item PPE → READY/NOT READY)
│   ├── camera_feed.py           # Camera abstraction (USB, WiFi, video)
│   ├── qr_scanner_opencv.py     # QR decoding with OpenCV
│   ├── ui_overlay.py            # Modern OpenCV overlay (PIL TrueType fonts)
│   ├── reporter.py              # HTTP reporter → backend API
│   ├── excel_reporter.py        # Local Excel report writer
│   ├── qr_generator.py          # QR ID card generator
│   ├── config.py                # Central configuration
│   └── ppe_model_v8.pt          # Trained YOLOv8 PPE model weights
│
├── backend/                     # Flask REST + WebSocket API
│   ├── app.py                   # Flask app factory + Socket.IO setup
│   ├── database.py              # SQLAlchemy init
│   ├── models.py                # DB models (CheckLog + LatestStatus)
│   └── routes/
│       ├── checks.py            # POST /api/report, GET /api/checks, etc.
│       └── dashboard.py         # GET /api/stats, /api/trend, /api/departments
│
├── frontend/                    # React/Vite dashboard
│   ├── src/
│   │   ├── App.jsx              # Root component + Socket.IO connection
│   │   ├── pages/
│   │   │   └── Dashboard.jsx    # Main dashboard page
│   │   ├── components/
│   │   │   ├── StatCards.jsx     # KPI cards (checks, ready, not ready, total)
│   │   │   ├── EmployeeTable.jsx # Live employee status table
│   │   │   ├── TrendChart.jsx   # 24-hour trend chart (Recharts)
│   │   │   ├── DepartmentChart.jsx # Department compliance chart
│   │   │   ├── CheckHistory.jsx # Recent check entries
│   │   │   └── LiveAlert.jsx    # Real-time toast notification
│   │   └── lib/
│   │       └── exportDashboardExcel.js # Client-side Excel export
│   └── package.json
│
├── employee_data/               # Employee registry
│   ├── employees.json           # Employee metadata (name, dept, role)
│   └── qr_cards/                # Generated QR ID card images
│
├── reports/                     # Auto-generated reports
│   └── employee_safety.xlsx     # Excel safety report (updated by AI station)
│
└── requirements.txt             # Python dependencies
```

---

## 🔧 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **AI / CV** | Python 3, OpenCV, Ultralytics YOLOv8, ByteTrack, Pillow, NumPy |
| **Backend** | Flask, Flask-SocketIO, Flask-SQLAlchemy, SQLite |
| **Frontend** | React 19, Vite 7, Recharts, Socket.IO Client, SheetJS (xlsx), Tailwind CSS |
| **Reporting** | openpyxl (server-side Excel), SheetJS (client-side Excel export) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ with pip
- Node.js 18+ with npm
- A camera source (webcam, USB mobile via DroidCam/Iriun, or WiFi IP camera)

### 1. Clone the Repository

```bash
git clone https://github.com/SUJALVAIDYA05/Industriguard-AI.git
cd Industriguard-AI
```

### 2. Set Up Python Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 3. Start the Backend

```bash
cd backend
python app.py
```

The backend starts on `http://localhost:5000`. On first run it will create the SQLite database at `backend/instance/industriguard.db`.

### 4. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

The dashboard opens at `http://localhost:5173`.

### 5. Run the AI Station

```bash
cd ai
python main_ai.py
```

> **Tip:** Edit `ai/config.py` to switch camera mode (`webcam`, `usb_mobile`, `wifi`, `video`) and adjust model/performance settings.

---

## ⚙️ Configuration (`ai/config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `CAMERA_MODE` | `"usb_mobile"` | Camera source: `webcam`, `usb_mobile`, `usb_tether`, `wifi`, `video` |
| `USB_CAMERA_INDEX` | `1` | Device index for USB cameras (0 = laptop, 1 = external) |
| `MODEL_PATH` | `"ppe_model_v8.pt"` | Path to the trained YOLOv8 weights |
| `USE_BYTE_TRACK` | `True` | Enable multi-person tracking via ByteTrack |
| `INFERENCE_EVERY_N_FRAMES` | `3` | Run YOLO every N frames (higher = faster FPS) |
| `INFERENCE_IMG_SIZE` | `480` | Input resolution for inference (lower = faster) |
| `PPE_FRAMES_NEEDED` | `10` | Frames to collect before making a final PPE decision |
| `RESULT_DISPLAY_SECONDS` | `5` | Seconds to show the result before resetting |
| `BACKEND_URL` | `"http://localhost:5000"` | Backend API URL |

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/report` | Receive a check result from the AI station |
| `GET` | `/api/stats` | Dashboard summary stats (today's checks, ready %, PPE violations) |
| `GET` | `/api/checks?limit=N` | Recent check history |
| `GET` | `/api/employees/status` | Latest status for all employees |
| `GET` | `/api/employees/<id>` | Single employee status + history |
| `GET` | `/api/trend` | 24-hour hourly trend data |
| `GET` | `/api/departments` | Department-wise compliance breakdown |
| `GET` | `/api/health` | Service health check |

**WebSocket Event:** `check_update` — Emitted on every new check result for real-time dashboard updates.

---

## 🔍 How It Works

1. **Scan** — A worker holds their QR ID card in front of the camera.
2. **Identify** — The QR scanner decodes the employee ID and looks it up in `employees.json`.
3. **Countdown** — A 5-second preparation timer gives the worker time to stand in position.
4. **Detect** — The YOLOv8 model runs across 10 frames, detecting all 5 PPE items per person.
5. **Decide** — Majority voting across frames determines final compliance. All 5 items present → **READY**, any missing → **NOT READY**.
6. **Report** — Results are saved to Excel, sent to the backend API, and broadcast via WebSocket to the live dashboard.
7. **Display** — The camera overlay shows a detailed PPE table, and the frontend dashboard updates in real-time with the employee's safety status and percentage.

---

## 🛡️ Safety Percentage Calculation

The safety percentage is calculated as:

$$\text{Safety \%} = \frac{\text{PPE items detected}}{5} \times 100$$

| Items Detected | Safety % |
|---------------|----------|
| 5/5 (all PPE) | 100% ✅ READY |
| 4/5 | 80% ❌ NOT READY |
| 3/5 | 60% ❌ NOT READY |
| 2/5 | 40% ❌ NOT READY |
| 1/5 | 20% ❌ NOT READY |
| 0/5 | 0% ❌ NOT READY |

> Only **100% compliance** (all 5 items) grants READY status.

---

## 📝 Current Status

### ✅ Implemented
- Full 5-item PPE detection pipeline (helmet, vest, gloves, goggles, boots)
- Multi-person ByteTrack tracking with QR-to-person association
- State machine workflow: SCANNING → COUNTDOWN → CHECKING → DISPLAYING
- Modern OpenCV overlay with glassmorphism and TrueType font rendering
- Flask backend with SQLite persistence, REST API, and WebSocket events
- React dashboard with stat cards, employee table, trend/department charts, check history, and live alerts
- Client-side and server-side Excel report generation
- Configurable camera modes (USB, WiFi, webcam, video file)

### 🔮 Planned / Future Improvements
- Multi-camera support with dedicated station IDs
- Authentication and role-based access control
- Docker containerization and CI/CD pipeline
- Automated test suite
- Production deployment hardening (monitoring, error recovery)
- PPE model fine-tuning for improved accuracy

---

## 📄 License

This project is for educational and demonstration purposes.
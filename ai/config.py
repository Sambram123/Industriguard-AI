# ═══════════════════════════════════════════════════════════
#   IndustriGuard AI — Central Configuration
#   Edit this file when your network changes
# ═══════════════════════════════════════════════════════════

# ── Camera Settings ────────────────────────────────────────
# Options:
#   0                              → Laptop webcam (fallback)
#   "http://192.168.x.x:8080/video" → Mobile camera (primary)
#   "video.mp4"                    → Recorded video (testing)

CAMERA_SOURCE = 0

# Set to True to use webcam instead (quick fallback)
USE_WEBCAM_FALLBACK = False

# ── Backend Settings ───────────────────────────────────────
BACKEND_URL = "http://localhost:5000"

# ── AI Model Settings ──────────────────────────────────────
MODEL_PATH = "yolo11n.pt"  # Replace with PPE model later

# ── Tracking Settings ───────────────────────────────────────
# Enables Ultralytics ByteTrack for stable person IDs across frames.
USE_BYTE_TRACK = True

# ── Performance Settings ────────────────────────────────────
# Run the heavy YOLO+tracking step only every N frames.
# (Higher = faster, but overlays update less frequently.)
INFERENCE_EVERY_N_FRAMES = 3

# Reduce the input size for faster inference. (Typical: 480 or 640)
# Set to None to use original frame size.
INFERENCE_IMG_SIZE = 480

# Turn off extra detector box drawing (saves CPU/GPU and avoids clutter)
DRAW_DETECTOR_BOXES = False

# ── Logging ────────────────────────────────────────────────
# Reduce terminal spam for better performance/readability.
VERBOSE_LOGS = False

# ── File Paths ─────────────────────────────────────────────
EMPLOYEES_FILE = "../employee_data/employees.json"
REPORT_PATH    = "../reports/employee_safety.xlsx"

# ── System Settings ────────────────────────────────────────
# Seconds to display result before resetting for next worker
RESULT_DISPLAY_SECONDS = 5

# How many frames to analyze for PPE confirmation
PPE_FRAMES_NEEDED = 10

# Camera ID shown in logs
CAMERA_ID = "CAM-01"
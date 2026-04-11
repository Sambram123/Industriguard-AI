import cv2
import time
import os
import sys

from config import (
    BACKEND_URL,
    MODEL_PATH,
    EMPLOYEES_FILE,
    REPORT_PATH,
    RESULT_DISPLAY_SECONDS,
    PPE_FRAMES_NEEDED,
    CAMERA_ID
)

from camera_feed    import CameraFeed
from qr_scanner_opencv import QRScanner  # Using OpenCV QR detector (no pyzbar)
from ppe_detector   import PPEDetector
from safety_status  import SafetyStatus
from excel_reporter import ExcelReporter
from reporter       import Reporter

# ── Startup ────────────────────────────────────────────────────────
print("\n" + "="*55)
print("   IndustriGuard AI — QR + PPE Safety Check System")
print("="*55 + "\n")

camera   = CameraFeed()                              # reads from config
scanner  = QRScanner(employees_file=EMPLOYEES_FILE)
detector = PPEDetector(model_path=MODEL_PATH)
safety   = SafetyStatus()
reporter = ExcelReporter(report_path=REPORT_PATH)
reporter_backend = Reporter(backend_url=BACKEND_URL)

cam_info = camera.get_info()
print(f"\n[Camera] Type   : {cam_info['type']}")
print(f"[Camera] Source : {cam_info['source']}")
print(f"[Camera] Size   : {cam_info['width']}x{cam_info['height']}")
print(f"[Camera] FPS    : {cam_info['fps']}")

print("\n[System] All modules ready.\n")
print("HOW TO USE:")
print("  1. Worker holds QR ID card toward camera")
print("  2. System scans QR → identifies employee")
print("  3. System checks PPE (helmet, vest)")
print("  4. Shows READY / NOT READY on screen")
print("  5. Result saved to Excel report")
print("\nPress Q to quit.\n")
print("-" * 55)

# ── State Machine ──────────────────────────────────────────────────
#
#  SCANNING   → waiting for QR code
#  CHECKING   → QR found, running PPE check
#  DISPLAYING → showing result, countdown to reset
#  RESET      → clear state, back to SCANNING
#
STATE             = "SCANNING"
current_employee  = None
current_status    = None
result_timer      = None
ppe_check_frames  = 0
ppe_results_pool  = []   # Collect results over multiple frames

# ADD THESE TWO LINES:
countdown_timer   = None
COUNTDOWN_SECONDS = 5

while True:
    frame = camera.get_frame()
    if frame is None:
        print("[Main] No frame received. Exiting.")
        break

    h, w = frame.shape[:2]

    # ── Top instruction banner ─────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 60, 120), -1)
    cv2.putText(
        frame,
        "IndustriGuard AI — Safety Check Station",
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75, (255, 255, 255), 2
    )

    # ══════════════════════════════════════════════════════════════
    # STATE: SCANNING — Wait for QR code
    # ══════════════════════════════════════════════════════════════
    if STATE == "SCANNING":

        # Show instruction
        cv2.putText(
            frame,
            "Please show your QR ID Card to the camera",
            (w // 2 - 250, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 0), 2
        )

        # Scanning border animation (yellow box in center)
        box_x1, box_y1 = w // 2 - 150, h // 2 - 150
        box_x2, box_y2 = w // 2 + 150, h // 2 + 100
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2),
                      (0, 255, 255), 2)
        cv2.putText(
            frame, "[ SCAN AREA ]",
            (box_x1 + 30, box_y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 255), 1
        )

        # Try to scan QR
        employee = scanner.scan_frame(frame)
        frame    = scanner.draw_qr_overlay(frame, employee)

    if employee and STATE == "SCANNING":
        current_employee = employee
        ppe_check_frames = 0
        ppe_results_pool = []
        countdown_timer  = time.time()
        STATE = "COUNTDOWN"
        scanner.reset()   # stops scanner from re-triggering

 # ══════════════════════════════════════════════════════════════
    # STATE: COUNTDOWN — Professional 5 second prep timer
    # ══════════════════════════════════════════════════════════════
    elif STATE == "COUNTDOWN":

        elapsed   = time.time() - countdown_timer
        remaining = COUNTDOWN_SECONDS - int(elapsed)

        # Dark overlay — top banner
        cv2.rectangle(frame, (0, 55), (w, 115), (10, 10, 30), -1)
        cv2.putText(
            frame,
            f"Welcome, {current_employee['name']}",
            (w // 2 - cv2.getTextSize(f"Welcome, {current_employee['name']}", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] // 2, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (200, 200, 255), 2
        )
        cv2.putText(
            frame,
            f"{current_employee['department']} | {current_employee['id']}",
            (w // 2 - cv2.getTextSize(f"{current_employee['department']} | {current_employee['id']}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] // 2, 107),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (120, 120, 180), 1
        )

        # Circle in center
        cx, cy = w // 2, h // 2
        cv2.circle(frame, (cx, cy), 90, (10, 10, 30), -1)       # fill
        cv2.circle(frame, (cx, cy), 90, (0, 180, 255), 3)        # outer ring
        cv2.circle(frame, (cx, cy), 75, (0, 100, 180), 1)        # inner ring

        # Countdown number — perfectly centered in circle
        if remaining > 0:
            count_text = str(remaining)
            font_scale = 4
            thickness  = 6
        else:
            count_text = "GO!"
            font_scale = 2
            thickness  = 4

        (tw, th), baseline = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = cx - tw // 2
        ty = cy + th // 2

        cv2.putText(
            frame, count_text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (0, 220, 255), thickness
        )

        # Message below circle
        if remaining > 0:
            msg = f"The PPE Scan Begins In  {remaining}  Second{'s' if remaining != 1 else ''}"
        else:
            msg = "Stand Still For PPE Scan"

        msg_w = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
        cv2.putText(
            frame, msg,
            (w // 2 - msg_w // 2, cy + 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2
        )

        # Progress bar at bottom
        progress  = min(elapsed / COUNTDOWN_SECONDS, 1.0)
        bar_width = int(w * progress)
        cv2.rectangle(frame, (0, h - 10), (w, h),         (20, 20, 40),   -1)
        cv2.rectangle(frame, (0, h - 10), (bar_width, h), (0, 180, 255),  -1)

        # Transition
        if elapsed >= COUNTDOWN_SECONDS:
            STATE = "CHECKING"
            print(f"[Main] Countdown done → Starting PPE check for {current_employee['name']}")

            # ══════════════════════════════════════════════════════════════
    # STATE: CHECKING — QR found, now check PPE
    # ══════════════════════════════════════════════════════════════
    elif STATE == "CHECKING":

        # Show checking banner
        cv2.rectangle(frame, (0, 55), (w, 95), (20, 100, 20), -1)
        cv2.putText(
            frame,
            f"Checking PPE for: {current_employee['name']}  "
            f"({ppe_check_frames}/{PPE_FRAMES_NEEDED})",
            (15, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2
        )

        # Run PPE detection on this frame
        detections = detector.detect(frame)
        compliance = detector.check_ppe_compliance(detections)
        frame      = detector.draw_boxes(frame, detections)

        # Collect result
        ppe_results_pool.append(compliance)
        ppe_check_frames += 1

        # After enough frames, make final decision
        if ppe_check_frames >= PPE_FRAMES_NEEDED:

            # Majority vote across collected frames
            helmet_votes = sum(1 for r in ppe_results_pool if r["has_helmet"])
            vest_votes   = sum(1 for r in ppe_results_pool if r["has_vest"])

            final_compliance = {
                "has_helmet": helmet_votes >= PPE_FRAMES_NEEDED // 2,
                "has_vest":   vest_votes   >= PPE_FRAMES_NEEDED // 2,
                "missing":    []
            }
            if not final_compliance["has_helmet"]:
                final_compliance["missing"].append("Helmet")
            if not final_compliance["has_vest"]:
                final_compliance["missing"].append("Safety Vest")

            # Evaluate final status
            current_status = safety.evaluate(final_compliance)

            # Save to Excel
            reporter.update_employee(current_employee, current_status)
            # Send to backend
            reporter_backend.send_check_result(current_employee, current_status, camera_id=CAMERA_ID)

            result_timer = time.time()
            STATE = "DISPLAYING"
            print(f"[Main] Result → {current_status['status']}")
            
    # ══════════════════════════════════════════════════════════════
    # STATE: DISPLAYING — Show result, then reset
    # ══════════════════════════════════════════════════════════════
    elif STATE == "DISPLAYING":

        # Draw status overlay
        frame = safety.draw_status(frame, current_status, current_employee)

        # Countdown timer
        elapsed   = time.time() - result_timer
        remaining = int(RESULT_DISPLAY_SECONDS - elapsed)

        cv2.putText(
            frame,
            f"Next check in {remaining}s...",
            (w - 220, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (200, 200, 200), 1
        )

        # Excel saved confirmation
        cv2.putText(
            frame,
            "✓ Saved to Excel Report",
            (w - 250, h - 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 100), 2
        )

        # Auto reset after display time
        if elapsed >= RESULT_DISPLAY_SECONDS:
            STATE = "SCANNING"
            scanner.reset()
            current_employee = None
            current_status   = None
            print("\n[Main] Ready for next worker...\n" + "-"*55)

    # ── Show frame ─────────────────────────────────────────────────
    cv2.imshow("IndustriGuard AI — Safety Check", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n[Main] Shutting down...")
        break

camera.release()
print("[Main] System stopped.\n")
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
    CAMERA_ID,
    USE_BYTE_TRACK,
    INFERENCE_EVERY_N_FRAMES,
    INFERENCE_IMG_SIZE,
    DRAW_DETECTOR_BOXES,
    WORKER_INFO_PERSIST_SECONDS
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

# Tracking state for stable labels (used when USE_BYTE_TRACK=True)
# Employee labels persist briefly even if tracking/QR drops for a few frames.
track_employee = {}     # track_id -> emp_dict (sticky while track is alive)
track_last_seen = {}    # track_id -> time.time()
recent_workers = {}     # emp_id -> latest overlay/report info

# Backend/Excel reporting de-dupe (multi-person mode)
MULTI_REPORT_MIN_INTERVAL_SECONDS = 5.0
last_sent = {}  # employee_id -> {"status": str, "has_helmet": bool, "has_vest": bool, "t": float}

# Cache last expensive inference results (for smooth FPS)
frame_index = 0
cached_detections = []
cached_persons_compliance = []

def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0

def _bbox_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def _dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy

def _qr_poly_to_rect(qr_poly):
    pts = qr_poly.reshape(-1, 2)
    xs = pts[:, 0]
    ys = pts[:, 1]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

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

    # ── Multi-person overlay (QR → person bbox + PPE + safety%) ─────
    # This runs every frame and does NOT change the existing single-person
    # state machine below (scan → countdown → checking → display).
    try:
        qr_results = scanner.scan_frame_multi(frame)
        frame_index += 1

        should_infer = (frame_index % max(1, int(INFERENCE_EVERY_N_FRAMES)) == 0)
        if should_infer:
            detections = (
                detector.detect_with_tracks_fast(frame, imgsz=INFERENCE_IMG_SIZE)
                if USE_BYTE_TRACK
                else detector.detect(frame)
            )
            cached_detections = detections
            cached_persons_compliance = detector.per_person_compliance(detections)
        else:
            detections = cached_detections

        if DRAW_DETECTOR_BOXES and detections:
            frame = detector.draw_boxes(frame, detections)

        persons_compliance = cached_persons_compliance
        now = time.time()

        # Mark currently visible tracks and keep them alive for a short grace period
        # so worker info does not disappear immediately on brief dropouts.
        visible_track_ids = set()
        for pc in (persons_compliance or []):
            tid = pc["person_det"].get("track_id")
            if tid is not None:
                tid = int(tid)
                visible_track_ids.add(tid)
                track_last_seen[tid] = now

        for tid in list(track_last_seen.keys()):
            if tid not in visible_track_ids and (now - track_last_seen[tid]) > WORKER_INFO_PERSIST_SECONDS:
                track_last_seen.pop(tid, None)
                track_employee.pop(tid, None)

        # Associate QR -> tracked person using IoU-first, distance fallback
        persons = [pc for pc in (persons_compliance or []) if pc.get("person_det")]

        # IoU pass
        used_person_idxs = set()
        for r in qr_results:
            emp = r.get("employee")
            poly = r.get("bbox")
            if not emp or poly is None:
                continue

            qr_rect = _qr_poly_to_rect(poly)

            best_i = None
            best_iou = 0.0
            for i, pc in enumerate(persons):
                if i in used_person_idxs:
                    continue
                pb = pc["person_det"]["bbox"]
                iou = _bbox_iou(qr_rect, pb)
                if iou > best_iou:
                    best_iou = iou
                    best_i = i

            if best_i is not None and best_iou >= 0.05:
                tid = persons[best_i]["person_det"].get("track_id")
                if tid is not None:
                    track_employee[int(tid)] = emp
                    used_person_idxs.add(best_i)

        # Distance fallback for any QR not assigned via IoU
        for r in qr_results:
            emp = r.get("employee")
            poly = r.get("bbox")
            if not emp or poly is None:
                continue

            qr_rect = _qr_poly_to_rect(poly)
            qc = _bbox_center(qr_rect)

            best_i = None
            best_d = None
            for i, pc in enumerate(persons):
                if i in used_person_idxs:
                    continue
                tid = pc["person_det"].get("track_id")
                if tid is None:
                    continue
                d = _dist2(qc, _bbox_center(pc["person_det"]["bbox"]))
                if best_d is None or d < best_d:
                    best_d = d
                    best_i = i

            if best_i is not None:
                tid = persons[best_i]["person_det"].get("track_id")
                if tid is not None:
                    track_employee[int(tid)] = emp
                    used_person_idxs.add(best_i)

        # Draw per-person overlays using stable track_id
        for pc in persons:
            pb = pc["person_det"]["bbox"]
            x1, y1, x2, y2 = pb
            tid = pc["person_det"].get("track_id")

            comp = pc
            has_helmet = bool(comp.get("has_helmet"))
            has_vest = bool(comp.get("has_vest"))
            safety_pct = int(comp.get("safety_percentage") or 0)
            status = "READY" if (has_helmet and has_vest) else "NOT READY"

            emp = None
            if tid is not None and int(tid) in track_employee:
                emp = track_employee[int(tid)]

            if emp:
                color = (0, 200, 0) if status == "READY" else (0, 0, 255)
            else:
                color = (180, 180, 180)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            if emp:
                lines = [
                    f"{emp['name']} ({emp['id']})",
                    f"{emp.get('department','')} | {emp.get('role','')}",
                    f"Helmet: {'Yes' if has_helmet else 'No'}  Vest: {'Yes' if has_vest else 'No'}",
                    f"Safety: {safety_pct}%  Status: {status}",
                ]

                recent_workers[emp["id"]] = {
                    "name": emp["name"],
                    "id": emp["id"],
                    "department": emp.get("department", ""),
                    "role": emp.get("role", ""),
                    "has_helmet": has_helmet,
                    "has_vest": has_vest,
                    "safety_pct": safety_pct,
                    "status": status,
                    "last_seen": now,
                }

                # Permanent association + continuous reporting:
                # send to backend + update Excel when status changes or at a slow interval.
                emp_id = emp["id"]
                prev = last_sent.get(emp_id)
                should_send = False
                if prev is None:
                    should_send = True
                else:
                    changed = (
                        prev["status"] != status or
                        prev["has_helmet"] != has_helmet or
                        prev["has_vest"] != has_vest
                    )
                    if changed or (now - prev["t"]) >= MULTI_REPORT_MIN_INTERVAL_SECONDS:
                        should_send = True

                if should_send:
                    compliance = {
                        "has_helmet": has_helmet,
                        "has_vest": has_vest,
                        "missing": ([] if has_helmet else ["Helmet"]) + ([] if has_vest else ["Safety Vest"])
                    }
                    status_data = safety.evaluate(compliance)
                    status_data["safety_percentage"] = safety_pct
                    status_data["track_id"] = int(tid) if tid is not None else None

                    # Save/update local Excel (one row per employee)
                    reporter.update_employee(emp, status_data)
                    # Publish to backend -> WebSocket -> frontend
                    reporter_backend.send_check_result(emp, status_data, camera_id=CAMERA_ID)

                    last_sent[emp_id] = {
                        "status": status,
                        "has_helmet": has_helmet,
                        "has_vest": has_vest,
                        "t": now
                    }
            else:
                label_id = f"#{int(tid)}" if tid is not None else ""
                lines = [
                    f"Person {label_id}".strip(),
                    f"Helmet: {'Yes' if has_helmet else 'No'}  Vest: {'Yes' if has_vest else 'No'}",
                    f"Safety: {safety_pct}%",
                ]

            # Draw worker info exactly once, anchored to the right side of the bbox.
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.52
            thickness = 2
            pad_x = 8
            pad_y = 6
            line_gap = 6

            sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
            text_w = max((sz[0] for sz in sizes), default=0)
            text_h = max((sz[1] for sz in sizes), default=0)
            box_w = text_w + pad_x * 2
            box_h = (text_h * len(lines)) + (line_gap * (len(lines) - 1)) + pad_y * 2

            # Preferred placement: immediately to the right of the bbox.
            bx1 = x2 + 10
            by1 = y1

            # Clamp inside frame (fallback to left side if needed).
            if bx1 + box_w > (w - 10):
                bx1 = max(10, x1 - 10 - box_w)
            by1 = min(max(10, by1), max(10, h - 10 - box_h))
            bx2 = bx1 + box_w
            by2 = by1 + box_h

            # Background + border (border color matches bbox color).
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

            # Text lines (top-to-bottom).
            ty = by1 + pad_y + text_h
            for line in lines:
                cv2.putText(frame, line, (bx1 + pad_x, ty), font, scale, (255, 255, 255), thickness)
                ty += text_h + line_gap

        # Also draw QR overlays (helpful for debugging association)
        frame = scanner.draw_qr_overlay_multi(frame, qr_results)
    except Exception as e:
        # Keep the main loop resilient
        print(f"[Main] Multi-overlay error: {e}")

    # ── Top instruction banner ─────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 60, 120), -1)
    cv2.putText(
        frame,
        "IndustriGuard AI — Safety Check Station",
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75, (255, 255, 255), 2
    )

    # Put scanning instruction in the header (not in the middle of frame)
    if STATE == "SCANNING":
        cv2.putText(
            frame,
            "Show QR ID card to the camera",
            (w - 340, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )

    # ══════════════════════════════════════════════════════════════
    # STATE: SCANNING — Wait for QR code
    # ══════════════════════════════════════════════════════════════
    if STATE == "SCANNING":

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
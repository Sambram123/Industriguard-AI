import cv2
import time
from config import (
    CAMERA_MODE,
    USB_CAMERA_INDEX,
    USB_TETHER_IP,
    USB_TETHER_PORT,
    WIFI_CAMERA_URL,
    VIDEO_FILE_PATH,
)


class CameraFeed:
    """
    Supports multiple camera connection modes:
      - usb_mobile  : Phone connected via USB (DroidCam, Iriun, Camo)
      - usb_tether  : Phone via USB tethering + IP Webcam app
      - wifi        : Phone via WiFi + IP Webcam app
      - webcam      : Laptop built-in webcam
      - video       : Recorded video file (testing)
    """

    def __init__(self, source=None):
        self.mode   = CAMERA_MODE
        self.source = source or self._resolve_source()
        self.is_stream = isinstance(self.source, str) and self.source.startswith("http")
        self.cap    = None
        self.reconnect_attempts = 0

        self._connect()

    # ── Resolve source from config mode ───────────────────────
    def _resolve_source(self):
        mode = CAMERA_MODE.lower().strip()

        if mode == "usb_mobile":
            # DroidCam / Iriun / Camo — phone appears as virtual webcam
            return USB_CAMERA_INDEX

        elif mode == "usb_tether":
            # IP Webcam app + USB tethering
            return f"http://{USB_TETHER_IP}:{USB_TETHER_PORT}/video"

        elif mode == "wifi":
            # IP Webcam app over WiFi
            return WIFI_CAMERA_URL

        elif mode == "webcam":
            return 0

        elif mode == "video":
            return VIDEO_FILE_PATH

        else:
            print(f"[CameraFeed] ⚠ Unknown CAMERA_MODE '{CAMERA_MODE}', falling back to webcam")
            return 0

    # ── Friendly name for logs ────────────────────────────────
    def _source_label(self):
        labels = {
            "usb_mobile": f"USB Mobile Camera (device {self.source})",
            "usb_tether": f"USB Tether + IP Webcam ({self.source})",
            "wifi":       f"WiFi Mobile Camera ({self.source})",
            "webcam":     "Laptop Webcam",
            "video":      f"Video File ({self.source})",
        }
        return labels.get(self.mode, str(self.source))

    # ── Connect ───────────────────────────────────────────────
    def _connect(self):
        print(f"[CameraFeed] Mode   : {self.mode}")
        print(f"[CameraFeed] Source : {self._source_label()}")
        print(f"[CameraFeed] Connecting...")

        # For USB mobile (DroidCam/Iriun), use DirectShow on Windows
        # for faster and more reliable connection
        if self.mode == "usb_mobile":
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.source)

        # Apply settings based on mode
        if self.is_stream:
            # HTTP streams need small buffer to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            # USB camera or webcam — request good resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self._raise_connection_error()

        print(f"[CameraFeed] ✅ Connected → {self._source_label()}")

    # ── Detailed error messages per mode ──────────────────────
    def _raise_connection_error(self):
        if self.mode == "usb_mobile":
            raise RuntimeError(
                f"\n[CameraFeed] ❌ Cannot open USB mobile camera (device {self.source})\n\n"
                f"   Troubleshoot:\n"
                f"   1. Install DroidCam or Iriun Webcam on your phone AND PC\n"
                f"   2. Connect phone to PC via USB cable\n"
                f"   3. Open the app on your phone and click 'Start'\n"
                f"   4. On PC, make sure the DroidCam/Iriun client is running\n"
                f"   5. Run  python find_cameras.py  to find the correct device index\n"
                f"   6. Update USB_CAMERA_INDEX in config.py (currently {USB_CAMERA_INDEX})\n"
                f"\n   Common indices: 0 = laptop webcam, 1 = USB phone camera\n"
            )

        elif self.mode == "usb_tether":
            raise RuntimeError(
                f"\n[CameraFeed] ❌ Cannot connect via USB tethering\n"
                f"   URL: {self.source}\n\n"
                f"   Troubleshoot:\n"
                f"   1. On phone: Settings → Connections → USB Tethering → ON\n"
                f"   2. Open IP Webcam app on phone and tap 'Start Server'\n"
                f"   3. Try opening {self.source} in your browser\n"
                f"   4. If IP is different, update USB_TETHER_IP in config.py\n"
            )

        elif self.mode == "wifi":
            raise RuntimeError(
                f"\n[CameraFeed] ❌ Cannot connect to WiFi camera\n"
                f"   URL: {self.source}\n\n"
                f"   Troubleshoot:\n"
                f"   1. Is IP Webcam app running on your phone?\n"
                f"   2. Are phone and laptop on the same WiFi?\n"
                f"   3. Open {self.source} in your browser first\n"
                f"   4. Update WIFI_CAMERA_URL in config.py\n"
            )

        else:
            raise RuntimeError(
                f"[CameraFeed] ❌ Cannot open camera (source={self.source})"
            )

    # ── Read frame ────────────────────────────────────────────
    def get_frame(self):
        """
        Returns a frame. Auto-reconnects for stream-based sources.
        """
        ret, frame = self.cap.read()

        if not ret:
            # For stream/network sources, try reconnecting
            if self.is_stream:
                print("[CameraFeed] ⚠ Frame dropped — attempting reconnect...")
                time.sleep(1)
                self._connect()
                self.reconnect_attempts += 1

                if self.reconnect_attempts > 10:
                    print("[CameraFeed] ❌ Too many failed reconnects. Exiting.")
                    return None

                ret, frame = self.cap.read()
                if ret:
                    self.reconnect_attempts = 0
                    print("[CameraFeed] ✅ Reconnected successfully")
                    return frame

            # For USB mobile, try reopening the device
            elif self.mode == "usb_mobile":
                print("[CameraFeed] ⚠ USB frame dropped — reopening device...")
                time.sleep(0.5)
                self.cap.release()
                self._connect()
                self.reconnect_attempts += 1

                if self.reconnect_attempts > 5:
                    print("[CameraFeed] ❌ USB camera not responding. Exiting.")
                    return None

                ret, frame = self.cap.read()
                if ret:
                    self.reconnect_attempts = 0
                    return frame

            return None

        self.reconnect_attempts = 0
        return frame

    # ── Camera info ───────────────────────────────────────────
    def get_info(self):
        """Returns camera info for display"""
        return {
            "source": self.source,
            "mode":   self.mode,
            "type":   self._source_label(),
            "width":  int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps":    int(self.cap.get(cv2.CAP_PROP_FPS))
        }

    # ── Release ───────────────────────────────────────────────
    def release(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("[CameraFeed] Camera released")
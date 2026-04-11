"""
Run this script to diagnose camera connection issues.
Usage: python diagnose_camera.py
"""
import cv2
import urllib.request
import socket
from config import (
    CAMERA_MODE,
    USB_CAMERA_INDEX,
    USB_TETHER_IP,
    USB_TETHER_PORT,
    WIFI_CAMERA_URL,
)


def check_network():
    print("\n── Network Check ─────────────────────────────")
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"   Laptop hostname : {hostname}")
    print(f"   Laptop IP       : {local_ip}")


def check_url_reachable(url):
    print("\n── URL Reachability Check ────────────────────")
    print(f"   Testing: {url}")
    try:
        urllib.request.urlopen(url, timeout=5)
        print("   ✅ URL is reachable")
        return True
    except Exception as e:
        print(f"   ❌ URL not reachable: {e}")
        return False


def check_opencv_connection(source):
    print("\n── OpenCV Connection Check ───────────────────")
    print(f"   Connecting to: {source}")

    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("   ❌ OpenCV cannot open camera")
        return False

    ret, frame = cap.read()
    if not ret:
        print("   ❌ OpenCV opened but cannot read frames")
        cap.release()
        return False

    h, w = frame.shape[:2]
    print(f"   ✅ OpenCV connected successfully")
    print(f"   ✅ Frame size: {w}x{h}")
    cap.release()
    return True


def run_diagnostics():
    print("\n" + "=" * 55)
    print("  IndustriGuard Camera Diagnostics")
    print("=" * 55)

    print(f"\n   Camera Mode: {CAMERA_MODE}")

    if CAMERA_MODE == "usb_mobile":
        print(f"   USB Camera Index: {USB_CAMERA_INDEX}")
        print("\n── USB Mobile Camera Check ───────────────────")
        check_opencv_connection(USB_CAMERA_INDEX)

    elif CAMERA_MODE == "usb_tether":
        url = f"http://{USB_TETHER_IP}:{USB_TETHER_PORT}/video"
        print(f"   USB Tether URL: {url}")
        check_network()
        base_url = f"http://{USB_TETHER_IP}:{USB_TETHER_PORT}"
        if check_url_reachable(base_url):
            check_opencv_connection(url)
        else:
            print("\n   Possible fixes:")
            print("   → Enable USB Tethering on your phone")
            print("   → Open IP Webcam and tap 'Start Server'")
            print(f"   → Update USB_TETHER_IP in config.py (currently {USB_TETHER_IP})")

    elif CAMERA_MODE == "wifi":
        print(f"   WiFi Camera URL: {WIFI_CAMERA_URL}")
        check_network()
        parts    = WIFI_CAMERA_URL.split("/")
        base_url = "/".join(parts[:3])
        if check_url_reachable(base_url):
            check_opencv_connection(WIFI_CAMERA_URL)
        else:
            print("\n   Possible fixes:")
            print("   → Make sure IP Webcam is running on your phone")
            print("   → Make sure phone and laptop are on same WiFi")
            print("   → Update WIFI_CAMERA_URL in config.py")

    elif CAMERA_MODE == "webcam":
        print("   Using laptop webcam (index 0)")
        check_opencv_connection(0)

    elif CAMERA_MODE == "video":
        from config import VIDEO_FILE_PATH
        print(f"   Video file: {VIDEO_FILE_PATH}")
        check_opencv_connection(VIDEO_FILE_PATH)

    print("\n" + "=" * 55 + "\n")


if __name__ == "__main__":
    run_diagnostics()
"""
IndustriGuard AI — Camera Finder Utility
=========================================
Run this script to find the correct device index for your
USB-connected mobile camera (DroidCam, Iriun, Camo, etc.)

Usage:  python find_cameras.py

It will scan device indices 0–9 and show which ones are available.
Use the correct index in config.py → USB_CAMERA_INDEX
"""

import cv2


def find_cameras():
    print("\n" + "=" * 55)
    print("  IndustriGuard AI — Camera Finder")
    print("=" * 55)
    print("\n  Scanning for available cameras (index 0–9)...\n")

    found = []

    for index in range(10):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                backend = cap.getBackendName()

                print(f"  ✅ Index {index}  →  {w}x{h} @ {fps}fps  (backend: {backend})")
                found.append(index)
            else:
                print(f"  ⚠  Index {index}  →  Opens but cannot read frames")
            cap.release()
        else:
            # Don't print anything for indices that don't open
            pass

    print()

    if not found:
        print("  ❌ No cameras detected!\n")
        print("  If using DroidCam / Iriun / Camo:")
        print("    1. Connect your phone to PC via USB cable")
        print("    2. Open the app on your phone")
        print("    3. Open the PC client (DroidCam Client / Iriun Webcam)")
        print("    4. Make sure it shows 'Connected' in the PC client")
        print("    5. Run this script again\n")
    else:
        print(f"  Found {len(found)} camera(s): {found}")
        print()

        if len(found) == 1:
            print(f"  → Set USB_CAMERA_INDEX = {found[0]} in config.py")
        elif len(found) >= 2:
            print(f"  → Index 0 is usually your laptop webcam")
            print(f"  → Index {found[1]} is likely your USB phone camera")
            print(f"  → Set USB_CAMERA_INDEX = {found[1]} in config.py")

        print()
        # Offer to preview
        print("  Would you like to preview a camera? (Enter index or 'n' to skip)")
        choice = input("  > ").strip()

        if choice.isdigit() and int(choice) in found:
            preview_camera(int(choice))

    print("=" * 55 + "\n")


def preview_camera(index):
    """Opens a preview window for the given camera index"""
    print(f"\n  Opening preview for camera index {index}...")
    print("  Press Q to close the preview.\n")

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"  ❌ Cannot open camera {index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  ⚠ Cannot read frame")
            break

        # Add label
        cv2.putText(
            frame,
            f"Camera Index: {index}  |  Press Q to close",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )

        cv2.imshow(f"Camera Preview — Index {index}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  Preview closed.")


if __name__ == "__main__":
    find_cameras()

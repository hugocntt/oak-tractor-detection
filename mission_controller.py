"""Top-level entry point for the autonomous tractor.

Launches the OAK-D detector and the MAVLink bridge as parallel threads.
The detector pushes DetectionResult objects into a shared queue every frame.
The bridge drains the queue and sends RC overrides to the Pixhawk over UART.

Usage:
    python mission_controller.py [--port /dev/ttyAMA0] [--baud 57600] [--no-bridge]

--no-bridge: run detector only (useful when no Pixhawk is connected, e.g. on your Mac)
"""
import argparse
import queue
import threading
import sys

from smart_detector import run as run_detector
from mavlink_bridge import MavlinkBridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",      default="/dev/ttyAMA0", help="Pixhawk UART port")
    parser.add_argument("--baud",      default=57600, type=int, help="UART baud rate")
    parser.add_argument("--no-bridge", action="store_true",    help="Skip MAVLink bridge (detector only)")
    args = parser.parse_args()

    result_queue: queue.Queue = queue.Queue(maxsize=10)

    bridge = None
    if not args.no_bridge:
        bridge = MavlinkBridge(result_queue, port=args.port, baud=args.baud)
        bridge.start()

    detector_thread = threading.Thread(
        target=run_detector,
        kwargs={"result_queue": result_queue},
        daemon=True,
        name="detector",
    )
    detector_thread.start()
    print("[Mission] Detector started. Press 'q' in the video window to quit.")

    try:
        detector_thread.join()
    except KeyboardInterrupt:
        print("\n[Mission] Interrupted.")
    finally:
        if bridge:
            bridge.stop()
        print("[Mission] Shutdown complete.")
        sys.exit(0)


if __name__ == "__main__":
    main()

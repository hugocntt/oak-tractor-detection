"""Standalone Pixhawk link test — confirms the Pi can hear the Pixhawk.

Run this BEFORE mission_controller.py to prove the serial link works in
isolation. It uses the exact same port and baud as the real MAVLink bridge
(imported from mavlink_bridge) so the two can never drift out of sync.

Usage:
    python heartbeat_test.py                       # uses defaults (/dev/ttyAMA0 @ 57600)
    python heartbeat_test.py --port /dev/ttyUSB0   # e.g. if connected over USB
    python heartbeat_test.py --baud 921600         # override baud if you changed it

Success looks like:
    Connected! Heartbeat from system 1, component 1

If it hangs on "Waiting for heartbeat...":
    - baud mismatch  -> SERIAL2_BAUD in QGroundControl must match (57 = 57600)
    - TX/RX swapped  -> swap the two data wires on the Pi GPIO
    - wrong port     -> check `ls /dev/ttyAMA*`
"""
import argparse
import sys

from pymavlink import mavutil

from mavlink_bridge import SERIAL_PORT, BAUD_RATE


def main():
    parser = argparse.ArgumentParser(description="Test the Pi <-> Pixhawk MAVLink link.")
    parser.add_argument("--port", default=SERIAL_PORT, help="serial port to the Pixhawk")
    parser.add_argument("--baud", default=BAUD_RATE, type=int, help="baud rate (must match SERIAL2_BAUD)")
    parser.add_argument("--timeout", default=15, type=int, help="seconds to wait before giving up")
    args = parser.parse_args()

    print(f"Opening {args.port} @ {args.baud} baud...")
    master = mavutil.mavlink_connection(args.port, baud=args.baud)

    print(f"Waiting for heartbeat (up to {args.timeout}s)...")
    hb = master.wait_heartbeat(timeout=args.timeout)

    if hb is None:
        print("\nNo heartbeat received.")
        print("  - Check SERIAL2_BAUD in QGroundControl matches this baud (57 = 57600).")
        print("  - Try swapping the TX/RX wires on the Pi.")
        print("  - Confirm the port exists:  ls /dev/ttyAMA*")
        sys.exit(1)

    print(f"\nConnected! Heartbeat from system {master.target_system}, "
          f"component {master.target_component}")
    sys.exit(0)


if __name__ == "__main__":
    main()

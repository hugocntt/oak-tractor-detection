"""MAVLink bridge — translates DetectionResult into Pixhawk RC overrides.

Normal operation: Pixhawk runs its ArduRover mission uninterrupted.
On STOP: channel 3 (throttle) is overridden to neutral (PWM_STOP).
On GO:   override is released (PWM 0 = hand control back to ArduRover).

The override is resent every 20 ms so the Pixhawk keeps it active.
If this process dies, the Pixhawk times out and resumes its mission on its own.
When an obstacle clears, a RESUME_DELAY_S grace period is enforced before GO.
"""
import time
import queue
import threading
from typing import Optional

from pymavlink import mavutil
from detection_result import DetectionResult

SERIAL_PORT   = "/dev/ttyAMA0"   # UART port to Pixhawk (change to /dev/ttyUSB0 if using USB)
BAUD_RATE     = 921600
OVERRIDE_HZ   = 10               # 10 Hz = every 100 ms
RESUME_DELAY_S = 2.0             # seconds path must be clear before resuming
PWM_STOP      = 1686             # throttle neutral — tractor holds position
PWM_RELEASE   = 0               # 0 = release override, ArduRover takes back control


class MavlinkBridge:
    def __init__(self, result_queue: queue.Queue, port: str = SERIAL_PORT, baud: int = BAUD_RATE):
        self.q = result_queue
        self.port = port
        self.baud = baud
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="mavlink-bridge")

    def start(self):
        self._thread.start()
        print(f"[MAVLink] Bridge started on {self.port} @ {self.baud}")

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=3)
        print("[MAVLink] Bridge stopped.")

    def _run(self):
        master = mavutil.mavlink_connection(self.port, baud=self.baud)
        print("[MAVLink] Waiting for heartbeat...")
        master.wait_heartbeat()
        print(f"[MAVLink] Connected — system {master.target_system} component {master.target_component}")

        current_action = "GO"
        clear_since: Optional[float] = None
        interval = 1.0 / OVERRIDE_HZ

        while not self._stop_event.is_set():
            # Drain queue — only the latest result matters
            latest: Optional[DetectionResult] = None
            try:
                while True:
                    latest = self.q.get_nowait()
            except queue.Empty:
                pass

            if latest is not None:
                if latest.action == "STOP":
                    current_action = "STOP"
                    clear_since = None
                else:
                    if current_action == "STOP":
                        # Path just cleared — start grace period
                        if clear_since is None:
                            clear_since = time.time()
                            print("[MAVLink] Path clear — waiting 2s before resuming...")
                        elif time.time() - clear_since >= RESUME_DELAY_S:
                            current_action = "GO"
                            clear_since = None
                            print("[MAVLink] Resuming mission.")
                    # if already GO, nothing to do

            if current_action == "STOP":
                self._send_override(master, PWM_STOP)
            else:
                self._send_override(master, PWM_RELEASE)

            time.sleep(interval)

    def _send_override(self, master, ch3_pwm: int):
        # Only override channel 3 (throttle). All others = 0 (no override).
        master.mav.rc_channels_override_send(
            master.target_system,
            master.target_component,
            0, 0, ch3_pwm, 0, 0, 0, 0, 0,
        )

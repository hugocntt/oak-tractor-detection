# Raspberry Pi Startup Guide — ZEAT Obstacle Detection

This file explains exactly what to run in the terminal on the Raspberry Pi to get
the full obstacle detection and MAVLink pipeline working. Run every step in order.

---

## Step 1 — Bring up the ethernet interface (camera network)

The OAK-4D Pro camera connects via PoE ethernet, not WiFi. The Pi needs a static
IP on the ethernet interface every time it boots.

```bash
sudo ip addr add 169.254.1.10/16 dev eth0
sudo ip link set eth0 up
```

**Expected output:** No output = good. An error saying "Address already assigned"
also means the IP is already set (fine, continue).

---

## Step 2 — Verify the camera is detected

```bash
python3 -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
```

**Expected output:**
```
[DeviceInfo(name=169.254.x.x, deviceId=XXXXXXXXX, X_LINK_GATE, X_LINK_TCP_IP, X_LINK_RVC4, X_LINK_SUCCESS)]
```

If the list is empty `[]`, the camera is not visible. Check:
- PoE switch is powered and has lights on the relevant ports
- Ethernet cables are firmly plugged in (Pi → switch, camera → switch)
- Camera LED is blue (powered)
- Re-run Step 1

---

## Step 3 — Verify the Pixhawk connection (MAVLink heartbeat)

The Pixhawk must be powered on before running this. It communicates with the Pi
via UART on `/dev/ttyAMA10` (Pi 5 specific — not ttyAMA0).

```bash
python3 heartbeat_test.py
```

**Expected output:**
```
waiting for heartbeat...
connected! system: 1
```

If it hangs indefinitely, check:
- Pixhawk is powered on (LEDs flashing)
- 3-wire UART cable is connected (TELEM2 on Pixhawk → GPIO pins 8, 10, 6 on Pi)
- TX/RX are not swapped — if it hangs, swap yellow and blue wires and try again
- Pixhawk parameters are set: `SERIAL2_PROTOCOL=2`, `SERIAL2_BAUD=921600`, `SYSID_MYGCS=255`

---

## Step 4a — Run the detector only (no Pixhawk needed)

Useful for testing the camera and YOLO detection without the tractor.

```bash
python3 mission_controller.py --no-bridge
```

**Expected output:**
```
Downloading yolov6-nano from Luxonis model zoo (first run only)...
Connected to OAK-4-PRO-W | USB: UNKNOWN
Model: yolov6-nano (running on Myriad X VPU)
Confidence threshold: 0.80
Press 'q' to quit.

[ GO ]         objects=0  obstacles=0  closest=999.0m (none)  fps=31
```

A window will open showing:
- **Left panel:** Live RGB feed with YOLO bounding boxes and distances
- **Right panel:** Top-down safety radar with range arcs
- **Bottom bar:** Green `PATH CLEAR  GO` or red `STOP  <object> at <d>m`

Put your hand or an object in front of the camera — the bar should turn red and
show `STOP`. Remove it — after ~0.6s of clear frames the bar turns green again.

Press `q` in the video window to quit.

---

## Step 4b — Run the full pipeline (detector + Pixhawk bridge)

Only run this when the Pixhawk heartbeat test (Step 3) has confirmed a connection.

```bash
python3 mission_controller.py
```

**Expected output:**
```
[MAVLink] Bridge started on /dev/ttyAMA10 @ 57600
Downloading yolov6-nano from Luxonis model zoo (first run only)...
[Mission] Detector started. Press 'q' in the video window to quit.
[MAVLink] Waiting for heartbeat...
Connected to OAK-4-PRO-W | USB: UNKNOWN
Model: yolov6-nano (running on Myriad X VPU)
[MAVLink] Connected — system 1 component 0
Press 'q' to quit.

[STOP][DEPTH]  objects=0  obstacles=0  closest=999.0m (none)  fps=31
```

When an object is detected:
- The terminal shows `[STOP]`
- The Pixhawk receives a throttle override to neutral (PWM 1686)
- The tractor holds position

When the path is clear:
- After 20 consecutive clear frames (~0.6s), the terminal shows `[ GO ]`
- The Pixhawk override is released and ArduRover resumes control

Press `Ctrl+C` to stop the pipeline cleanly.

---

## Useful notes

**The YOLO model is cached after first download.** Subsequent runs work offline — 
no internet needed. Only the very first run requires a WiFi or hotspot connection.

**Baud rate:** The Pixhawk TELEM2 port is configured at 921600. The Pi connects at
the same rate. Do not change `SERIAL2_BAUD` in QGroundControl without also
updating `BAUD_RATE` in `mavlink_bridge_2.py`.

**Pi 5 UART port:** The correct port is `/dev/ttyAMA10`. Earlier Pi models use
`/dev/ttyAMA0`. Using the wrong port produces a "no such file or directory" error.

**Static IP persistence:** The `sudo ip addr add` command in Step 1 does not
survive a reboot. Run it every session before starting the detector. To make it
permanent, ask for the `/etc/dhcpcd.conf` setup instructions.

**Tuning the stop sensitivity:**
- `EMERGENCY_M` in `smart_detector.py` — depth trigger distance (default 1.5m)
- `DANGER_M` — YOLO trigger distance (default 3.0m)
- `RESUME_CLEAR_FRAMES` — consecutive clear frames before GO (default 20, ~0.6s)
- `MIN_VALID_FRAC` — minimum valid depth pixel fraction to trust a clear reading (default 0.10)
- `RESUME_DELAY_S` in `mavlink_bridge.py` — additional bridge-side grace before releasing override (default 2.0s)

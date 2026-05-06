# Smart Obstacle Detection — OAK-D + Pixhawk Autonomous Tractor

> A two-layer obstacle-avoidance pipeline that lets an ArduRover-driven tractor
> see the world, recognise what's in front of it, and stop safely on its own.

A **Luxonis OAK-D PoE** camera runs **YOLOv6-nano** on its on-device VPU and
fuses every detection with stereo depth to recover real-world 3D coordinates.
The host computer adds a class-agnostic depth fallback (so the tractor still
stops for a tree stump or a fence post that COCO has never heard of), then
issues `STOP` / `GO` commands to a **Pixhawk** flight controller via a
**MAVLink** RC channel override.

Normal operation: the Pixhawk runs its pre-loaded ArduRover mission untouched.
On `STOP`, the host overrides channel 3 (throttle) to neutral, holding the
rover in place. On `GO`, the override is released and ArduRover resumes the
mission on its own.

---

## Highlights

- **On-device inference** — YOLOv6-nano runs on the OAK-D's Myriad X VPU. The
  host only consumes results, so even a Raspberry Pi keeps up.
- **3D from a single camera** — `SpatialDetectionNetwork` fuses each YOLO
  detection with stereo depth to give X, Y, Z in millimetres.
- **Two redundant safety layers** — a class-aware YOLO layer plus a
  class-agnostic depth fallback. If one misses, the other catches.
- **Fail-open design** — if the host crashes, the Pixhawk's RC override times
  out within ~3 s and the rover resumes its mission unaided. No silent failures.
- **Live HUD** — RGB feed with bounding boxes and 3D coordinates on the left,
  a top-down safety radar (range arcs, FOV cone, plotted obstacles) on the right.
- **IoU tracker with EMA smoothing** — bounding boxes don't jitter, distances
  don't flicker, and short YOLO dropouts don't break the decision loop.

---

## Architecture

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                        OAK-D PoE Camera                         │
 │   ┌──────────────┐         ┌────────────────┐                   │
 │   │ RGB Sensor   │────────►│ YOLOv6-nano    │                   │
 │   └──────────────┘         │ (Myriad X VPU) │                   │
 │   ┌──────────────┐         └───────┬────────┘                   │
 │   │ Stereo Pair  │────────►        │                            │
 │   │ (depth)      │   ┌─────────────▼─────────────┐              │
 │   └──────────────┘   │ SpatialDetectionNetwork   │              │
 │                      │ (3D coords per detection) │              │
 │                      └─────────────┬─────────────┘              │
 └────────────────────────────────────┼────────────────────────────┘
                                      │ frames + detections + depth
                                      ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                    Host: smart_detector.py                      │
 │  ┌─────────────────┐    ┌──────────────────┐    ┌────────────┐  │
 │  │ IoU tracker +   │    │ Depth fallback   │    │ Decision   │  │
 │  │ EMA smoothing   │───►│ (p10 of zone)    │───►│ STOP / GO  │  │
 │  └─────────────────┘    └──────────────────┘    └─────┬──────┘  │
 │                                                       │         │
 │              ┌────────────────────────────────────────┘         │
 │              ▼                                                  │
 │   DetectionResult ──► thread-safe queue.Queue                   │
 └──────────────────────────────────┬──────────────────────────────┘
                                    ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                    Host: mavlink_bridge.py                      │
 │   • Drains queue at 50 Hz (only the latest result matters)      │
 │   • STOP → ch3 override = 1686 µs (neutral, holds position)     │
 │   • GO   → ch3 override = 0     (release, ArduRover resumes)    │
 │   • 2-second clearance grace before resuming after a STOP       │
 └──────────────────────────────────┬──────────────────────────────┘
                                    │ UART, MAVLink
                                    ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │              Pixhawk (ArduRover) — runs mission                 │
 └─────────────────────────────────────────────────────────────────┘
```

---

## Two-layer safety

The decision logic combines a **class-aware** layer (YOLO) and a
**class-agnostic** fallback (depth percentile):

| Layer        | What it sees                          | Trigger                               |
|--------------|---------------------------------------|---------------------------------------|
| YOLO         | Named obstacles (person, animal, …)   | Distance under `DANGER_M` (3.0 m)     |
| Depth p10    | Anything in the safety band           | 10th-percentile depth under `EMERGENCY_M` (1.5 m) |

The depth band looks at the **centre third** of the frame (the tractor's track
width) between **35 %** and **75 %** of frame height — this skips the sky and
the ground plane so the rover never falsely stops on its own shadow. The 10th
percentile (rather than the median) is sensitive enough to catch an object
covering only ~10 % of the zone, while staying robust against a few noisy
pixels.

Why a fallback? YOLO is trained on COCO. It knows people, dogs, cars, chairs —
but it has never seen a tree stump, a hay bale, or a fence post. The depth
layer doesn't care what an object **is**, only that something solid is close.

---

## Repository layout

```
smart_detector/
├── README.md                  ← this file
├── LICENSE
├── requirements.txt           ← Python dependencies
├── .gitignore
├── __init__.py
├── detection_result.py        ← DetectionResult dataclass (shared contract)
├── smart_detector.py          ← OAK-D pipeline + visualiser (perception)
├── mavlink_bridge.py          ← MAVLink RC override (actuation)
└── mission_controller.py      ← Entry point — launches both as threads
```

---

## Hardware

- **Luxonis OAK-D PoE** (or any OAK-D variant with stereo + RGB)
- **Pixhawk** flight controller running ArduRover firmware
- A host computer (Raspberry Pi 4/5, Jetson, or laptop) with:
  - Network or USB access to the OAK-D
  - UART or USB-serial connection to the Pixhawk

---

## Setup

```bash
# 1. Clone and enter the folder
cd smart_detector/

# 2. (Recommended) create a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

Dependencies (see `requirements.txt`):

- `depthai>=3.0` — Luxonis SDK (note: this project uses the **v3 API**)
- `opencv-python>=4.5` — visualiser
- `numpy>=1.20`
- `pymavlink>=2.4` — MAVLink bridge

The first run downloads the YOLOv6-nano model from the Luxonis model zoo
(roughly 10 MB) and caches it under `.depthai_cached_models/`.

---

## Usage

Run from inside the `smart_detector/` directory.

### Full pipeline (detector + Pixhawk bridge)

```bash
python mission_controller.py --port /dev/ttyAMA0 --baud 57600
```

Common Pixhawk ports:

- Raspberry Pi UART: `/dev/ttyAMA0` or `/dev/serial0`
- USB-serial:        `/dev/ttyUSB0` or `/dev/ttyACM0`

### Detector only (no Pixhawk connected)

Useful for development on a laptop:

```bash
python mission_controller.py --no-bridge
```

### Smart detector standalone

```bash
python smart_detector.py
```

Press `q` in the video window to quit.

---

## What you see on screen

- **Left panel — RGB + YOLO detections.** Bounding boxes coloured by threat
  level (green = clear, orange = danger, red = emergency). Each obstacle is
  labelled with class, distance, confidence, and X/Y/Z in millimetres. The
  depth-zone rectangle in the centre turns red when the depth fallback fires.
- **Right panel — top-down safety radar.** Range arcs at `EMERGENCY_M` and
  `DANGER_M`, a ±30° FOV cone, and one dot per tracked obstacle plotted by
  range and lateral offset. A cyan `STOP <d>m` marker appears when the depth
  layer triggers on an unrecognised object.
- **Bottom status bar.** Green `PATH CLEAR  GO` or red
  `STOP  <object> at <d>m`, plus live FPS and detection counts.

---

## Configuration

All tuneables live at the top of each module.

`smart_detector.py`:

| Constant                  | Default       | Meaning                                       |
|---------------------------|---------------|-----------------------------------------------|
| `MODEL_NAME`              | `yolov6-nano` | Model fetched from the Luxonis zoo            |
| `FPS`                     | 60            | Camera frame rate                             |
| `CONFIDENCE_THRESHOLD`    | 0.65          | Minimum YOLO confidence                       |
| `DEPTH_MIN_MM` / `MAX_MM` | 300 / 10000   | Depth clip range                              |
| `EMERGENCY_M`             | 1.5           | Depth fallback STOP threshold (m)             |
| `DANGER_M`                | 3.0           | YOLO STOP threshold (m)                       |
| `TRACK_SMOOTH`            | 0.4           | Bounding-box EMA (lower = smoother but laggier) |
| `TRACK_MAX_MISS`          | 6             | Frames a track survives without a detection   |
| `RESUME_GRACE_S`          | 2.0           | Detector-side clearance grace before GO       |

`mavlink_bridge.py`:

| Constant         | Default | Meaning                                          |
|------------------|---------|--------------------------------------------------|
| `OVERRIDE_HZ`    | 50      | Override resend rate (every 20 ms)               |
| `RESUME_DELAY_S` | 2.0     | Bridge-side clearance grace before releasing override |
| `PWM_STOP`       | 1686    | Channel 3 PWM (µs) for neutral throttle          |
| `PWM_RELEASE`    | 0       | Sentinel that releases the override              |

`OBSTACLE_CLASSES` is built from the COCO label map by removing airborne or
non-physical entries (frisbee, kite, …) and small handheld items (cup, bottle,
phone, …) that aren't relevant for an outdoor rover.

---

## Data contract

Every frame, the detector pushes a `DetectionResult` into the shared queue:

```python
DetectionResult(
    action="STOP",            # "STOP" or "GO"
    closest_dist_m=2.4,       # nearest obstacle distance in metres (999 = none)
    closest_obj="person",     # label of the nearest obstacle ("none" if clear)
    depth_triggered=False,    # True if the depth fallback (not YOLO) caused the STOP
    depth_zone_m=4.1,         # 10th-percentile depth in the safety band (999 = no data)
    timestamp=1715812800.42,  # epoch seconds when the result was produced
)
```

---

## Failure modes

- **Bridge process dies** → Pixhawk stops receiving overrides → the override
  times out (~3 s, ArduPilot's `RC_OVR_TIMEOUT`) and ArduRover resumes the
  mission unconditionally. This is the intended fail-open behaviour for a
  controller-level safety stop.
- **OAK-D loses connection** → the detector loop exits; the bridge keeps
  resending its last action until shutdown. The mission controller stops both
  threads cleanly on `Ctrl+C`.
- **Depth dropout** (no valid pixels in the band) → `depth_zone_m` stays at
  999 and the depth fallback is effectively disabled for that frame; YOLO
  remains the sole decision source.

---

## Implementation notes

- The detector and bridge run as **independent threads** sharing one
  `queue.Queue(maxsize=10)`. The bridge only consumes the **latest** result by
  draining the queue, so detector lag never queues stale STOPs.
- The override is resent at 50 Hz (every 20 ms), well under the Pixhawk's RC
  override timeout (~3 s).
- The 2-second resume grace is enforced **on both sides** (detector and
  bridge). Defence in depth: a flickering YOLO detection cannot bypass it on
  the detector side, and a stuck-`GO` queue cannot bypass it on the bridge
  side.
- The IoU tracker matches detections by class label and an IoU threshold of
  0.25, then EMA-smooths bounding boxes and distance. New tracks are created
  for unmatched detections; old tracks are dropped after `TRACK_MAX_MISS`
  frames without a match.

---

## License

See [LICENSE](LICENSE).

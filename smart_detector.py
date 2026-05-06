"""Smart obstacle detection using YOLOv6-nano + stereo depth on OAK-D.

This uses the camera's on-device neural network (Myriad X VPU) to detect
real objects (people, animals, vehicles, etc.) and combines each detection
with stereo depth to get its 3D position (X, Y, Z in mm).

Why this is better than pixel counting:
 - The neural network KNOWS what an object is (person, dog, car, chair...)
 - It ignores grass, ground, sky, walls — only detects real things
 - Each detection gets a 3D distance from stereo depth automatically
 - Runs entirely on the camera chip — your Mac just displays results

Pipeline:
  RGB Camera ──→ YOLOv6-nano (on Myriad X VPU)
  Stereo Depth ──→ fused with detections (SpatialDetectionNetwork)
  Output: labelled objects with X, Y, Z coordinates in mm

Usage:
    python smart_detector.py

Press 'q' to quit.
"""
import queue
import depthai as dai
import numpy as np
import cv2
import time
import math
from typing import Optional

from detection_result import DetectionResult

# ── Config ───────────────────────────────────────────────────────────────
MODEL_NAME = "yolov6-nano"          # auto-downloaded from Luxonis model zoo
FPS = 60                            # camera frame rate
TRACK_SMOOTH = 0.4                  # bbox EMA — lower = smoother but more lag (0–1)
TRACK_MAX_MISS = 6                  # frames to keep a track alive without a detection
CONFIDENCE_THRESHOLD = 0.65          # min confidence to count a detection
DEPTH_MIN_MM = 300                  # ignore closer than 30cm
DEPTH_MAX_MM = 10000                # ignore further than 10m

# Obstacle distance thresholds (metres)
EMERGENCY_M = 1.5                   # STOP
DANGER_M = 3.0                      # STOP threshold

# COCO classes — what YOLOv6-nano can detect
LABEL_MAP = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",
    "bus",            "train",      "truck",         "boat",          "traffic light",
    "fire hydrant",   "stop sign",  "parking meter", "bench",         "bird",
    "cat",            "dog",        "horse",         "sheep",         "cow",
    "elephant",       "bear",       "zebra",         "giraffe",       "backpack",
    "umbrella",       "handbag",    "tie",           "suitcase",      "frisbee",
    "skis",           "snowboard",  "sports ball",   "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",
    "wine glass",     "cup",        "fork",          "knife",         "spoon",
    "bowl",           "banana",     "apple",         "sandwich",      "orange",
    "broccoli",       "carrot",     "hot dog",       "pizza",         "donut",
    "cake",           "chair",      "sofa",          "pottedplant",   "bed",
    "diningtable",    "toilet",     "tvmonitor",     "laptop",        "mouse",
    "remote",         "keyboard",   "cell phone",    "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",
    "vase",           "scissors",   "teddy bear",    "hair drier",    "toothbrush",
]

# Everything COCO detects is a potential obstacle except airborne/abstract things
NON_OBSTACLES = {
    "aeroplane", "kite", "frisbee", "skis", "snowboard", "sports ball",
    "baseball bat", "baseball glove", "surfboard", "tennis racket",
    "wine glass", "fork", "knife", "spoon", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "remote", "keyboard", "cell phone", "mouse", "toothbrush", "hair drier",
    "scissors", "book", "vase", "clock", "laptop", "tvmonitor",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "bottle", "cup", "bowl",
}
OBSTACLE_CLASSES = {label for label in LABEL_MAP if label not in NON_OBSTACLES}

# Output window
OUTPUT_W = 1280
OUTPUT_H = 720


class Track:
    _next_id = 0

    def __init__(self, det: dict):
        self.id = Track._next_id
        Track._next_id += 1
        self.label = det["label"]
        self.bbox = list(det["bbox"])       # [x1, y1, x2, y2] floats for EMA
        self.dist_m = det["dist_m"]
        self.x_mm = det["x_mm"]
        self.y_mm = det["y_mm"]
        self.z_mm = det["z_mm"]
        self.confidence = det["confidence"]
        self.is_obstacle = det["is_obstacle"]
        self.misses = 0

    def update(self, det: dict):
        a = TRACK_SMOOTH
        bx = det["bbox"]
        self.bbox = [
            a * bx[0] + (1 - a) * self.bbox[0],
            a * bx[1] + (1 - a) * self.bbox[1],
            a * bx[2] + (1 - a) * self.bbox[2],
            a * bx[3] + (1 - a) * self.bbox[3],
        ]
        self.dist_m = a * det["dist_m"] + (1 - a) * self.dist_m
        self.x_mm = det["x_mm"]
        self.y_mm = det["y_mm"]
        self.z_mm = det["z_mm"]
        self.confidence = det["confidence"]
        self.misses = 0

    def as_detection(self) -> dict:
        x1, y1, x2, y2 = (int(v) for v in self.bbox)
        level, colour = classify_distance(self.dist_m)
        return {
            "label": self.label,
            "confidence": self.confidence,
            "dist_m": self.dist_m,
            "level": level,
            "colour": colour,
            "bbox": (x1, y1, x2, y2),
            "is_obstacle": self.is_obstacle,
            "x_mm": self.x_mm,
            "y_mm": self.y_mm,
            "z_mm": self.z_mm,
        }


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def match_detections(tracks: list, dets: list):
    """Greedy IoU matching. Returns (matched pairs, unmatched dets, unmatched tracks)."""
    matched, used_t, used_d = [], set(), set()
    for di, d in enumerate(dets):
        best_iou, best_ti = 0.0, -1
        for ti, t in enumerate(tracks):
            if ti in used_t or t.label != d["label"]:
                continue
            iou = _iou(t.bbox, d["bbox"])
            if iou > best_iou:
                best_iou, best_ti = iou, ti
        if best_iou > 0.25:
            matched.append((best_ti, di))
            used_t.add(best_ti)
            used_d.add(di)
    unmatched_d = [i for i in range(len(dets)) if i not in used_d]
    unmatched_t = [i for i in range(len(tracks)) if i not in used_t]
    return matched, unmatched_d, unmatched_t


def classify_distance(dist_m: float):
    """Return (level, colour) for a distance."""
    if dist_m < EMERGENCY_M:
        return "EMERGENCY", (0, 0, 255)
    if dist_m < DANGER_M:
        return "DANGER", (0, 128, 255)
    return "CLEAR", (0, 200, 0)


def build_pipeline():
    pipeline = dai.Pipeline()

    # ── RGB camera ──
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A, sensorFps=FPS)

    # ── Stereo depth ──
    stereo = pipeline.create(dai.node.StereoDepth).build(autoCreateCameras=True)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    # ── Spatial Detection Network (YOLO + depth fusion on-device) ──
    model_desc = dai.NNModelDescription()
    model_desc.model = MODEL_NAME
    nn = pipeline.create(dai.node.SpatialDetectionNetwork).build(cam, stereo, model_desc)
    nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    nn.setDepthLowerThreshold(DEPTH_MIN_MM)
    nn.setDepthUpperThreshold(DEPTH_MAX_MM)
    nn.setBoundingBoxScaleFactor(0.5)
    nn.setNumInferenceThreads(2)
    nn.setNumNCEPerInferenceThread(2)

    # ── Output queues ──
    det_queue = nn.out.createOutputQueue(maxSize=4, blocking=False)
    rgb_queue = nn.passthrough.createOutputQueue(maxSize=4, blocking=False)
    depth_queue = nn.passthroughDepth.createOutputQueue(maxSize=4, blocking=False)

    return pipeline, det_queue, rgb_queue, depth_queue


def draw_status_bar(canvas, action, worst_level, closest_obj, closest_dist, det_count, fps):
    bar_h = 120
    bar_y = OUTPUT_H - bar_h

    bg = (0, 0, 180) if action == "STOP" else (0, 120, 0)
    cv2.rectangle(canvas, (0, bar_y), (OUTPUT_W, OUTPUT_H), bg, -1)

    text = f"STOP  {closest_obj} at {closest_dist:.1f}m" if action == "STOP" else "PATH CLEAR  GO"

    cv2.putText(canvas, text, (20, bar_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    stats = (f"Detections: {det_count}    "
             f"Threat: {worst_level}    "
             f"Closest: {closest_dist:.2f}m ({closest_obj})    "
             f"FPS: {fps:.0f}")
    cv2.putText(canvas, stats, (20, bar_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)


def run(result_queue: Optional[queue.Queue] = None):
    print(f"Downloading {MODEL_NAME} from Luxonis model zoo (first run only)...")
    pipeline, det_queue, rgb_queue, depth_queue = build_pipeline()
    pipeline.start()

    device = pipeline.getDefaultDevice()
    print(f"Connected to {device.getDeviceName()} | USB: {device.getUsbSpeed().name}")
    print(f"Model: {MODEL_NAME} (running on Myriad X VPU)")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Press 'q' to quit.\n")

    prev_time = time.time()
    fps = 0.0
    tracks: list[Track] = []
    clear_since: Optional[float] = None   # timestamp when path first became clear
    RESUME_GRACE_S = 2.0                  # seconds of clear path before GO

    while pipeline.isRunning():
        in_det = det_queue.get()
        in_rgb = rgb_queue.get()
        in_depth = depth_queue.get()

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 0.001))
        prev_time = now

        frame = in_rgb.getCvFrame()
        depth_frame = in_depth.getFrame()
        detections = in_det.detections

        h, w = frame.shape[:2]

        # ── Build raw detections from YOLO output ──
        raw_dets = []
        for det in detections:
            label_idx = det.label
            label = LABEL_MAP[label_idx] if label_idx < len(LABEL_MAP) else f"class_{label_idx}"
            z_mm = det.spatialCoordinates.z
            dist_m = z_mm / 1000.0
            if dist_m <= 0:
                continue
            x1 = int(det.xmin * w)
            y1 = int(det.ymin * h)
            x2 = int(det.xmax * w)
            y2 = int(det.ymax * h)
            raw_dets.append({
                "label": label,
                "confidence": det.confidence,
                "dist_m": dist_m,
                "bbox": (x1, y1, x2, y2),
                "is_obstacle": label in OBSTACLE_CLASSES,
                "x_mm": det.spatialCoordinates.x,
                "y_mm": det.spatialCoordinates.y,
                "z_mm": z_mm,
            })

        # ── Update tracker ──
        matched, unmatched_d, unmatched_t = match_detections(tracks, raw_dets)
        for ti, di in matched:
            tracks[ti].update(raw_dets[di])
        for di in unmatched_d:
            tracks.append(Track(raw_dets[di]))
        for ti in unmatched_t:
            tracks[ti].misses += 1
        tracks = [t for t in tracks if t.misses <= TRACK_MAX_MISS]

        # ── Build display detections from smoothed tracks ──
        detection_info = [t.as_detection() for t in tracks]

        worst_level = "CLEAR"
        worst_severity = 0
        closest_dist = 999.0
        closest_obj = "none"
        obstacle_count = 0
        severity_map = {"CLEAR": 0, "DANGER": 1, "EMERGENCY": 2}

        for d in detection_info:
            if not d["is_obstacle"]:
                continue
            level = d["level"]
            if severity_map[level] > worst_severity:
                worst_severity = severity_map[level]
                worst_level = level
            if d["dist_m"] < closest_dist:
                closest_dist = d["dist_m"]
                closest_obj = d["label"]
                obstacle_count += 1

        # Decide action (YOLO layer)
        if worst_level in ("EMERGENCY", "DANGER"):
            action = "STOP"
            clear_since = None
        else:
            action = "GO"

        # ── Depth safety layer (class-agnostic fallback for unrecognised objects) ──
        # Zone: centre third (tractor track width), rows 25–60% (skip sky + ground).
        # Metric: 10th percentile of valid depths — catches any object covering ~10%
        # of the zone without being fooled by a few noisy pixels.
        dh, dw = depth_frame.shape[:2]
        row_start = int(dh * 0.35)
        row_end   = int(dh * 0.75)
        col_start = dw // 3
        col_end   = 2 * dw // 3
        zone = depth_frame[row_start:row_end, col_start:col_end]
        valid = zone[(zone >= DEPTH_MIN_MM) & (zone <= DEPTH_MAX_MM)]
        depth_triggered = False
        depth_zone_m = 999.0
        if valid.size > 0:
            depth_zone_m = float(np.percentile(valid, 10)) / 1000.0
            if depth_zone_m < EMERGENCY_M:
                action = "STOP"
                depth_triggered = True
                clear_since = None

        # Apply 2-second grace period: only GO after path has been clear long enough
        if action == "GO":
            if clear_since is None:
                clear_since = now
            if now - clear_since < RESUME_GRACE_S:
                action = "STOP"

        # Push result to MAVLink bridge (if running)
        if result_queue is not None:
            result_queue.put(DetectionResult(
                action=action,
                closest_dist_m=closest_dist,
                closest_obj=closest_obj,
                depth_triggered=depth_triggered,
                depth_zone_m=depth_zone_m,
            ))

        # Console
        trigger_tag = "[DEPTH]" if depth_triggered else "       "
        tag = "STOP" if action == "STOP" else " GO "
        print(f"[{tag}]{trigger_tag}  objects={len(detection_info)}  obstacles={obstacle_count}  "
              f"closest={closest_dist:.1f}m ({closest_obj})  fps={fps:.0f}  ", end='\r')

        # ── Build output frame ──
        canvas = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)

        # Left: RGB with detections
        rgb_display = frame.copy()
        for d in detection_info:
            x1, y1, x2, y2 = d["bbox"]
            colour = d["colour"] if d["is_obstacle"] else (180, 180, 180)
            thickness = 2 if d["is_obstacle"] else 1

            cv2.rectangle(rgb_display, (x1, y1), (x2, y2), colour, thickness)

            # Label with distance
            text = f"{d['label']} {d['dist_m']:.1f}m ({d['confidence']:.0%})"
            text_y = max(y1 - 8, 15)
            cv2.putText(rgb_display, text, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

            # Show 3D coords for obstacles
            if d["is_obstacle"]:
                coord_text = f"X:{d['x_mm']:.0f} Y:{d['y_mm']:.0f} Z:{d['z_mm']:.0f}mm"
                cv2.putText(rgb_display, coord_text, (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)

        # Draw depth check zone — centre third, obstacle-height band
        zone_x1 = w // 3
        zone_x2 = 2 * w // 3
        zone_y1 = int(h * 0.35)
        zone_y2 = int(h * 0.75)
        zone_col = (0, 0, 220) if depth_triggered else (0, 180, 0)
        cv2.rectangle(rgb_display, (zone_x1, zone_y1), (zone_x2, zone_y2), zone_col, 2)
        depth_label = f"depth zone: {depth_zone_m:.1f}m (stop <{EMERGENCY_M}m)" if depth_zone_m < 900 else "depth zone: --"
        cv2.putText(rgb_display, depth_label, (zone_x1 + 4, zone_y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, zone_col, 1)

        cv2.putText(rgb_display, "RGB + YOLO Detections", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Right: top-down safety radar panel
        view_h = OUTPUT_H - 120
        view_w = OUTPUT_W // 2
        radar = np.zeros((view_h, view_w, 3), dtype=np.uint8)

        # Background grid
        radar[:] = (15, 15, 15)
        cx, cy = view_w // 2, view_h - 30   # tractor position at bottom-centre
        max_dist_px = view_h - 60           # pixels = DANGER_M metres

        def m_to_px(dist_m):
            return int(dist_m / DANGER_M * max_dist_px)

        # Concentric range arcs
        for dist_m, label in [(EMERGENCY_M, f"{EMERGENCY_M:.1f}m"), (DANGER_M, f"{DANGER_M:.1f}m STOP")]:
            r = m_to_px(dist_m)
            colour = (0, 0, 200) if dist_m == EMERGENCY_M else (0, 100, 200)
            cv2.ellipse(radar, (cx, cy), (r, r), 0, 180, 360, colour, 1)
            cv2.putText(radar, label, (cx + 6, cy - r + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)

        # Tractor FOV cone (±30°)
        fov_r = max_dist_px + 20
        for angle_deg in (-30, 30):
            angle_rad = math.radians(angle_deg - 90)
            ex = int(cx + fov_r * math.cos(angle_rad))
            ey = int(cy + fov_r * math.sin(angle_rad))
            cv2.line(radar, (cx, cy), (ex, ey), (50, 50, 50), 1)

        # Fill emergency zone red, danger zone orange (semi-transparent via overlay)
        overlay = radar.copy()
        cv2.ellipse(overlay, (cx, cy), (m_to_px(EMERGENCY_M), m_to_px(EMERGENCY_M)),
                    0, 180, 360, (0, 0, 80), -1)
        cv2.ellipse(overlay, (cx, cy), (m_to_px(DANGER_M), m_to_px(DANGER_M)),
                    0, 180, 360, (0, 40, 80), -1)
        cv2.ellipse(overlay, (cx, cy), (m_to_px(EMERGENCY_M), m_to_px(EMERGENCY_M)),
                    0, 180, 360, (15, 15, 15), -1)
        radar = cv2.addWeighted(overlay, 0.4, radar, 0.6, 0)

        # Plot obstacles as dots
        for d in detection_info:
            if not d["is_obstacle"]:
                continue
            dist_m = d["dist_m"]
            x_mm = d["x_mm"]
            # map lateral offset: x_mm / (DANGER_M*1000) scaled to half-panel-width
            lateral_px = int(x_mm / (DANGER_M * 1000) * (view_w // 2))
            dot_x = np.clip(cx + lateral_px, 5, view_w - 5)
            dot_y = np.clip(cy - m_to_px(dist_m), 5, view_h - 5)
            dot_colour = d["colour"]
            cv2.circle(radar, (dot_x, dot_y), 10, dot_colour, -1)
            cv2.putText(radar, f"{d['label']}", (dot_x + 12, dot_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, dot_colour, 1)
            cv2.putText(radar, f"{dist_m:.1f}m", (dot_x + 12, dot_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # Depth layer dot (unknown object) — plotted at actual measured distance
        if depth_triggered:
            dot_y_depth = np.clip(cy - m_to_px(depth_zone_m), 5, view_h - 5)
            cv2.circle(radar, (cx, dot_y_depth), 12, (0, 200, 255), -1)
            cv2.putText(radar, f"STOP {depth_zone_m:.1f}m", (cx + 15, dot_y_depth + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

        # Tractor icon at bottom
        cv2.rectangle(radar, (cx - 12, cy - 10), (cx + 12, cy + 10), (180, 180, 180), -1)
        cv2.putText(radar, "TRACTOR", (cx - 28, cy + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # Title
        cv2.putText(radar, "SAFETY RADAR", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Depth zone median distance indicator
        zone_colour = (0, 0, 220) if depth_triggered else (0, 180, 0)
        zone_text = f"Depth zone median: {depth_zone_m:.1f}m" if depth_zone_m < 900 else "Depth zone: no data"
        cv2.putText(radar, zone_text, (10, view_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, zone_colour, 1)

        # Combine into canvas
        rgb_resized = cv2.resize(rgb_display, (view_w, view_h))

        canvas[0:view_h, 0:view_w] = rgb_resized
        canvas[0:view_h, view_w:OUTPUT_W] = radar
        cv2.line(canvas, (view_w, 0), (view_w, view_h), (100, 100, 100), 2)

        # Status bar
        draw_status_bar(canvas, action, worst_level, closest_obj, closest_dist,
                        len(detection_info), fps)

        cv2.imshow("Smart Obstacle Detection (YOLO + Depth)", canvas)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\n\nStopped.")


if __name__ == "__main__":
    run()

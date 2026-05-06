"""Shared data contract between the detector and the MAVLink bridge.

Every frame, smart_detector.run() pushes one DetectionResult into a
queue.Queue. mavlink_bridge.MavlinkBridge drains that queue and translates
the action into RC channel overrides for the Pixhawk.
"""
from dataclasses import dataclass, field
import time


@dataclass
class DetectionResult:
    action: str             # "GO" or "STOP"
    closest_dist_m: float   # distance to nearest obstacle (999 = none)
    closest_obj: str        # label of nearest obstacle ("none" if clear)
    depth_triggered: bool   # True if depth fallback (not YOLO) caused the STOP
    depth_zone_m: float     # median depth in the check zone (999 = no data)
    timestamp: float = field(default_factory=time.time)

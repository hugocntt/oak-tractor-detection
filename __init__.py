"""Smart obstacle detection for an OAK-D + Pixhawk autonomous tractor.

This directory holds runnable scripts. Run them from inside this folder:

    python mission_controller.py            # full pipeline (detector + bridge)
    python mission_controller.py --no-bridge  # detector only (no Pixhawk)
    python smart_detector.py                # detector only, standalone
"""
__version__ = "0.1.0"

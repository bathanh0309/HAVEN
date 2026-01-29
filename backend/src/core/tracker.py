
import cv2
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class Tracker:
    """
    Wrapper for Ultralytics Tracking (ByteTrack/BoT-SORT).
    """
    def __init__(self, model_path, tracker_type="bytetrack"):
        self.model = YOLO(model_path)
        self.tracker_type = tracker_type + ".yaml"  # bytetrack.yaml or botsort.yaml
        logger.info(f"Initialized Tracker: {self.tracker_type}")

    def update(self, frame):
        """
        Run tracking on frame.
        Returns: results object from YOLO
        """
        # persist=True enables tracking
        results = self.model.track(
            frame, 
            persist=True, 
            tracker=self.tracker_type,
            verbose=False,
            # Config params can be passed here if needed
        )
        return results

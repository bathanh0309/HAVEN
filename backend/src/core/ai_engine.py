
import logging
import time
import numpy as np
import cv2
from typing import Dict, Any, Optional
from ultralytics import YOLO
from ..config.camera_config import CameraSettings

logger = logging.getLogger(__name__)

class AIEngine:
    """
    AI Processing Engine with Frame Skipping & Caching.
    
    Optimized for CPU usage:
    - Lazy Loading: Model only loads when first needed.
    - Frame Skipping: Inference runs only every N frames.
    - Caching: Reuse last detection result for skipped frames.
    """
    
    def __init__(self, config: CameraSettings):
        self.config = config
        self.model = None
        self.model_path = config.AI_MODEL_PATH
        
        # Performance Tuning
        self.skip_frames = 4  # Run AI every 4 frames (Target ~5-7 FPS inference on CPU)
        self.frame_counter = 0
        self.last_annotated_frame = None
        self.last_results = None
        self.last_inference_time = 0
    
    def load_model(self):
        if self.model: return
        try:
            logger.info(f"Loading AI Model: {self.model_path}")
            # Load model (YOLOv8 Pose)
            self.model = YOLO(self.model_path)
            logger.info("AI Model Loaded")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run inference if it's time, otherwise reuse old result logic.
        Returns: Annotated Frame (BGR Image)
        """
        if self.model is None:
            self.load_model()
            if self.model is None:
                return frame # Return raw frame if AI dead

        self.frame_counter += 1
        
        # ==========================
        # Frame Skipping Logic
        # ==========================
        should_run_ai = (self.frame_counter % self.skip_frames == 0)
        
        if should_run_ai:
            try:
                start_time = time.time()
                
                # Run Inference
                # conf=0.5: Filter low confidence
                # verbose=False: Reduce log spam
                # imgsz=640: Standard YOLO input size
                results = self.model(frame, verbose=False, conf=0.5, imgsz=640)
                
                self.last_results = results[0]
                self.last_inference_time = (time.time() - start_time) * 1000 # ms
                
                # Render results immediately
                # plot() returns a new numpy array with annotations
                self.last_annotated_frame = self.last_results.plot()
                
            except Exception as e:
                logger.error(f"Inference Error: {e}")
                return frame
        
        # ==========================
        # Result Caching Strategy
        # ==========================
        # If we have a cached annotated frame, use it?
        # PRO: Extremely fast for skipped frames (0ms latency cost).
        # CON: Video looks "jerky" if movement is fast because frame background is old.
        #
        # BETTER STRATEGY for "Visual Fluidity" + "Overlay Persistence":
        # We should Ideally draw the OLD boxes on the NEW frame. 
        # But results[0].plot() draws on the original image used for inference.
        #
        # For simplicity "Quick Win": Return the cached annotated frame? NO, that freezes video.
        # Correct approach: Return `frame` (NEW) but draw LAST known boxes.
        
        if self.last_results:
            try:
                # Use Ultralytics plotter to draw boxes on the NEW frame
                # result.plot(img=frame) allows drawing on custom image
                return self.last_results.plot(img=frame)
            except Exception as e:
                # Fallback to no annotation if drawing fails
                return frame
                
        return frame

# Singleton
_ai_instance = None
def get_ai_engine(config: CameraSettings = None) -> AIEngine:
    global _ai_instance
    if _ai_instance is None and config:
        _ai_instance = AIEngine(config)
    return _ai_instance

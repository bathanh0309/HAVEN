"""
StreamHub - Multi-camera registry and lifecycle manager
Singleton that manages multiple StreamWorker instances keyed by camera_id.
"""

import logging
from typing import Dict, Optional, List
from core.stream_worker import StreamWorker, CameraConfig

logger = logging.getLogger(__name__)


class StreamHub:
    """
    Multi-camera manager (singleton pattern).
    
    Responsibilities:
    - Register cameras with unique camera_id
    - Start/stop all workers
    - Provide lookup: camera_id  StreamWorker
    - Aggregate status from all cameras
    """
    
    def __init__(self):
        self._workers: Dict[str, StreamWorker] = {}
        self._ai_engine = None  # Set via set_ai_engine()
    
    def set_ai_engine(self, ai_engine):
        """Set the AI engine for all workers."""
        self._ai_engine = ai_engine
        logger.info("AI engine configured for StreamHub")
    
    def register_camera(self, config: CameraConfig) -> bool:
        """
        Register a new camera and create its worker.
        Does not start the worker yet (call start_all() later).
        
        Returns:
            bool: True if registered successfully, False if camera_id already exists
        """
        if config.camera_id in self._workers:
            logger.warning(f"Camera {config.camera_id} already registered")
            return False
        
        worker = StreamWorker(config, ai_engine=self._ai_engine)
        self._workers[config.camera_id] = worker
        
        logger.info(
            f"Registered camera: {config.camera_id}  {config.source} "
            f"(AI: {config.ai_enabled})"
        )
        return True
    
    def get_worker(self, camera_id: str) -> Optional[StreamWorker]:
        """Get worker by camera_id (returns None if not found)."""
        return self._workers.get(camera_id)
    
    def list_cameras(self) -> List[str]:
        """Get list of all registered camera IDs."""
        return list(self._workers.keys())
    
    def get_all_statuses(self) -> List[Dict]:
        """Get status from all cameras."""
        return [worker.get_status() for worker in self._workers.values()]
    
    def start_all(self):
        """Start all registered workers."""
        logger.info(f"Starting {len(self._workers)} camera workers...")
        for camera_id, worker in self._workers.items():
            try:
                worker.start()
            except Exception as e:
                logger.error(f"Failed to start {camera_id}: {e}")
    
    def stop_all(self):
        """Stop all workers gracefully."""
        logger.info(f"Stopping {len(self._workers)} camera workers...")
        for camera_id, worker in self._workers.items():
            try:
                worker.stop()
            except Exception as e:
                logger.error(f"Failed to stop {camera_id}: {e}")
    
    def remove_camera(self, camera_id: str) -> bool:
        """
        Remove a camera (stops worker if running).
        
        Returns:
            bool: True if removed, False if not found
        """
        worker = self._workers.get(camera_id)
        if not worker:
            return False
        
        worker.stop()
        del self._workers[camera_id]
        logger.info(f"Removed camera: {camera_id}")
        return True


# Global singleton instance
_stream_hub_instance: Optional[StreamHub] = None


def get_stream_hub() -> StreamHub:
    """Get or create the global StreamHub singleton."""
    global _stream_hub_instance
    if _stream_hub_instance is None:
        _stream_hub_instance = StreamHub()
    return _stream_hub_instance


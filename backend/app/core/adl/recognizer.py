"""
ADL Recognizer Interface
Supports both rule-based and temporal model approaches
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from .actions import ADLAction, ADLLabel
from .rule_based import RuleBasedRecognizer
# from .temporal_model import TemporalModelRecognizer  # Phase B


class ADLRecognizer(ABC):
    """
    Abstract base class for ADL recognizers
    """
    
    @abstractmethod
    def recognize(self, keypoints: np.ndarray, bbox: tuple, frame: np.ndarray) -> Optional[ADLAction]:
        """
        Recognize activity from pose keypoints
        
        Args:
            keypoints: (17, 3) array of (x, y, confidence)
            bbox: (x, y, w, h) bounding box
            frame: Current frame (optional, for context)
        
        Returns:
            ADLAction or None if no activity detected
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset internal state (for temporal models)"""
        pass
    
    @staticmethod
    def create(model_type: str = "rule_based", **kwargs) -> 'ADLRecognizer':
        """
        Factory method to create recognizer
        
        Args:
            model_type: "rule_based" or "temporal"
            **kwargs: Additional config
        
        Returns:
            ADLRecognizer instance
        """
        if model_type == "rule_based":
            return RuleBasedRecognizer(**kwargs)
        elif model_type == "temporal":
            # Phase B: ST-GCN implementation
            raise NotImplementedError("Temporal model not yet implemented")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

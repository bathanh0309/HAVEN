"""
Image processing utilities.
"""
import cv2
import numpy as np
from typing import Tuple, Optional


def compute_blur_score(image: np.ndarray) -> float:
    """
    Compute blur score using Laplacian variance.
    
    Args:
        image: Input image (BGR or grayscale)
    
    Returns:
        Blur score (higher = sharper)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Compute Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    return variance


def crop_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int], 
              padding: int = 0) -> Optional[np.ndarray]:
    """
    Crop bounding box region from image.
    
    Args:
        image: Input image
        bbox: Bounding box (x, y, w, h)
        padding: Extra padding around bbox (pixels)
    
    Returns:
        Cropped image or None if invalid
    """
    x, y, w, h = bbox
    h_img, w_img = image.shape[:2]
    
    # Add padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)
    
    # Check validity
    if x2 <= x1 or y2 <= y1:
        return None
    
    return image[y1:y2, x1:x2]


def resize_keep_aspect(image: np.ndarray, target_size: Tuple[int, int],
                        fill_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Resize image keeping aspect ratio with letterboxing.
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        fill_color: Color for letterbox padding
    
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create canvas
    canvas = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)
    
    # Paste resized image in center
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
    
    return canvas


def draw_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int],
              label: str = "", color: Tuple[int, int, int] = (0, 255, 0),
              thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box on image.
    
    Args:
        image: Input image
        bbox: Bounding box (x, y, w, h)
        label: Optional label text
        color: Box color (BGR)
        thickness: Line thickness
    
    Returns:
        Image with drawn bbox
    """
    x, y, w, h = bbox
    
    # Draw rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw background
        cv2.rectangle(image, (x, y - text_h - 10), (x + text_w, y), color, -1)
        
        # Draw text
        cv2.putText(image, label, (x, y - 5), font, font_scale, (255, 255, 255), font_thickness)
    
    return image


def draw_keypoints(image: np.ndarray, keypoints: np.ndarray,
                   color: Tuple[int, int, int] = (0, 255, 0),
                   radius: int = 3) -> np.ndarray:
    """
    Draw pose keypoints on image.
    
    Args:
        image: Input image
        keypoints: Keypoints array (17, 3) [x, y, confidence]
        color: Keypoint color (BGR)
        radius: Keypoint radius
    
    Returns:
        Image with drawn keypoints
    """
    # Define skeleton connections
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    # Draw skeleton connections
    for start_idx, end_idx in skeleton:
        start_kpt = keypoints[start_idx]
        end_kpt = keypoints[end_idx]
        
        # Check if both keypoints are visible
        if start_kpt[2] > 0.5 and end_kpt[2] > 0.5:
            start_point = (int(start_kpt[0]), int(start_kpt[1]))
            end_point = (int(end_kpt[0]), int(end_kpt[1]))
            cv2.line(image, start_point, end_point, color, 2)
    
    # Draw keypoints
    for kpt in keypoints:
        if kpt[2] > 0.5:  # Only draw visible keypoints
            center = (int(kpt[0]), int(kpt[1]))
            cv2.circle(image, center, radius, color, -1)
    
    return image


if __name__ == "__main__":
    # Test utilities
    import cv2
    
    # Test blur score
    test_img = cv2.imread("/path/to/test/image.jpg")
    blur_score = compute_blur_score(test_img)
    print(f"Blur score: {blur_score:.2f}")
    
    # Test crop
    bbox = (100, 100, 200, 300)
    cropped = crop_bbox(test_img, bbox, padding=10)
    if cropped is not None:
        print(f"Cropped shape: {cropped.shape}")

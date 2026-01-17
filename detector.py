
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
import time


class WildlifeDetector:
    # COCO dataset wildlife classes (class_id: name)
    WILDLIFE_CLASSES = {
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 
        18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
        22: 'zebra', 23: 'giraffe'
    }
    
    # Rarity tiers for alerting
    RARITY_TIERS = {
        'common': ['bird', 'cat', 'dog'],
        'interesting': ['horse', 'sheep', 'cow'],
        'rare': ['elephant', 'bear', 'zebra', 'giraffe']
    }
    
    def __init__(self, model_size: str = 'n', confidence: float = 0.5):
        """
        Initialize wildlife detector
        
        Args:
            model_size: 'n'(nano), 's'(small), 'm'(medium) - nano recommended
            confidence: Detection confidence threshold (0.0-1.0)
        """
        self.confidence = confidence
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.fps_history = []
        
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict], float]:
        """
        Detect wildlife in frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            annotated_frame: Frame with bounding boxes
            detections: List of detection dicts
            fps: Current frames per second
        """
        start_time = time.time()
        
        # Run inference
        results = self.model(frame, verbose=False)[0]
        
        # Filter wildlife only
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if class_id in self.WILDLIFE_CLASSES and conf >= self.confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                species = self.WILDLIFE_CLASSES[class_id]
                
                detections.append({
                    'species': species,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'rarity': self._get_rarity(species)
                })
        
        # Annotate frame
        annotated = self._draw_detections(frame.copy(), detections)
        
        # Calculate FPS
        fps = self._calculate_fps(time.time() - start_time)
        
        return annotated, detections, fps
    
    def _get_rarity(self, species: str) -> str:
        """Determine rarity tier of species"""
        for tier, animals in self.RARITY_TIERS.items():
            if species in animals:
                return tier
        return 'common'
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        colors = {
            'common': (100, 255, 100),    # Green
            'interesting': (255, 200, 0),  # Orange
            'rare': (0, 100, 255)          # Red
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = colors.get(det['rarity'], (255, 255, 255))
            
            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label background
            label = f"{det['species']} {det['confidence']:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            
            # Label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def _calculate_fps(self, frame_time: float) -> float:
        """Calculate rolling average FPS"""
        fps = 1.0 / max(frame_time, 0.001)
        self.fps_history.append(fps)
        
        # Keep last 30 frames
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        
        return np.mean(self.fps_history)
    
    def add_stats_overlay(self, frame: np.ndarray, fps: float, 
                         detection_count: int) -> np.ndarray:
        """Add professional stats overlay to frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Stats text
        stats = [
            f"FPS: {fps:.1f}",
            f"Detections: {detection_count}",
            f"Model: YOLOv8n"
        ]
        
        y_offset = 35
        for stat in stats:
            cv2.putText(frame, stat, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        return frame
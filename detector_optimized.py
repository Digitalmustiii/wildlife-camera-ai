"""
Optimized Wildlife Detector with Multiple Backends
Supports: YOLOv8 (default), ONNX Runtime, TensorRT
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import time
from pathlib import Path


class OptimizedWildlifeDetector:
    """Wildlife detector with optimized inference backends"""
    
    WILDLIFE_CLASSES = {
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 
        18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
        22: 'zebra', 23: 'giraffe'
    }
    
    RARITY_TIERS = {
        'common': ['bird', 'cat', 'dog'],
        'interesting': ['horse', 'sheep', 'cow'],
        'rare': ['elephant', 'bear', 'zebra', 'giraffe']
    }
    
    def __init__(self, model_path: str, backend: str = 'auto', confidence: float = 0.5):
        """
        Initialize optimized detector
        
        Args:
            model_path: Path to model (can be .pt, .onnx, or .engine)
            backend: 'auto', 'yolov8', 'onnx', 'onnx-int8', 'tensorrt'
            confidence: Detection confidence threshold
        """
        self.confidence = confidence
        self.backend = backend
        self.fps_history = []
        
        # Auto-detect backend if needed
        if backend == 'auto':
            self.backend = self._detect_best_backend(model_path)
        
        print(f"🔧 Loading model with backend: {self.backend.upper()}")
        
        # Initialize appropriate backend
        if self.backend == 'yolov8':
            self._init_yolov8(model_path)
        elif self.backend.startswith('onnx'):
            self._init_onnx(model_path)
        elif self.backend == 'tensorrt':
            self._init_tensorrt(model_path)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        print(f"✅ Backend loaded: {self.backend.upper()}")
    
    def _detect_best_backend(self, model_path: str) -> str:
        """Auto-detect best available backend"""
        base_path = Path(model_path).stem
        
        # Check for optimized models
        if Path(f"{base_path}_int8.onnx").exists():
            return 'onnx-int8'
        elif Path(f"{base_path}.onnx").exists():
            return 'onnx'
        elif Path(f"{base_path}.engine").exists():
            return 'tensorrt'
        else:
            return 'yolov8'
    
    def _init_yolov8(self, model_path: str):
        """Initialize YOLOv8 backend"""
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.input_size = 640
    
    def _init_onnx(self, model_path: str):
        """Initialize ONNX Runtime backend"""
        try:
            import onnxruntime as ort
            
            # Determine correct model path
            if self.backend == 'onnx-int8':
                onnx_path = model_path.replace('.pt', '_int8.onnx')
            else:
                onnx_path = model_path.replace('.pt', '.onnx')
            
            if not Path(onnx_path).exists():
                raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
            
            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            
            # Create session
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input/output details
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.input_size = self.input_shape[2]  # Assuming NCHW format
            
            print(f"✅ ONNX model loaded: {onnx_path}")
            print(f"   Providers: {self.session.get_providers()}")
            
        except ImportError:
            raise ImportError("ONNX Runtime not installed. Install: pip install onnxruntime")
    
    def _init_tensorrt(self, model_path: str):
        """Initialize TensorRT backend"""
        try:
            from ultralytics import YOLO
            
            engine_path = model_path.replace('.pt', '.engine')
            if not Path(engine_path).exists():
                raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
            
            # YOLOv8 can load TensorRT engines directly
            self.model = YOLO(engine_path, task='detect')
            self.input_size = 640
            
            print(f"✅ TensorRT engine loaded: {engine_path}")
            
        except Exception as e:
            raise RuntimeError(f"TensorRT initialization failed: {e}")
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict], float]:
        """
        Detect wildlife in frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            annotated_frame, detections, fps
        """
        start_time = time.time()
        
        # Run inference based on backend
        if self.backend == 'yolov8' or self.backend == 'tensorrt':
            detections = self._detect_yolov8(frame)
        else:  # ONNX
            detections = self._detect_onnx(frame)
        
        # Annotate frame
        annotated = self._draw_detections(frame.copy(), detections)
        
        # Calculate FPS
        fps = self._calculate_fps(time.time() - start_time)
        
        return annotated, detections, fps
    
    def _detect_yolov8(self, frame: np.ndarray) -> List[Dict]:
        """YOLOv8/TensorRT detection"""
        results = self.model(frame, verbose=False)[0]
        
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
        
        return detections
    
    def _detect_onnx(self, frame: np.ndarray) -> List[Dict]:
        """ONNX Runtime detection"""
        # Preprocess
        input_tensor = self._preprocess_onnx(frame)
        
        # Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Postprocess (YOLOv8 ONNX output format)
        detections = self._postprocess_onnx(outputs, frame.shape)
        
        return detections
    
    def _preprocess_onnx(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for ONNX inference"""
        # Resize
        img = cv2.resize(frame, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def _postprocess_onnx(self, outputs: List[np.ndarray], original_shape: Tuple) -> List[Dict]:
        """Postprocess ONNX outputs to detections"""
        # YOLOv8 ONNX output: [1, 84, 8400] (for COCO)
        # Format: [x, y, w, h, class_0_conf, class_1_conf, ..., class_79_conf]
        predictions = outputs[0][0]  # Remove batch dimension
        
        detections = []
        
        # Transpose to [8400, 84]
        predictions = predictions.T
        
        for pred in predictions:
            # Extract bbox and class confidences
            x, y, w, h = pred[:4]
            class_confs = pred[4:]
            
            # Get best class
            class_id = int(np.argmax(class_confs))
            conf = float(class_confs[class_id])
            
            # Filter by confidence and wildlife classes
            if class_id in self.WILDLIFE_CLASSES and conf >= self.confidence:
                # Convert to original image coordinates
                h_ratio = original_shape[0] / self.input_size
                w_ratio = original_shape[1] / self.input_size
                
                x1 = int((x - w/2) * w_ratio)
                y1 = int((y - h/2) * h_ratio)
                x2 = int((x + w/2) * w_ratio)
                y2 = int((y + h/2) * h_ratio)
                
                # Clip to frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(original_shape[1], x2)
                y2 = min(original_shape[0], y2)
                
                species = self.WILDLIFE_CLASSES[class_id]
                
                detections.append({
                    'species': species,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'rarity': self._get_rarity(species)
                })
        
        return detections
    
    def _get_rarity(self, species: str) -> str:
        """Determine rarity tier of species"""
        for tier, animals in self.RARITY_TIERS.items():
            if species in animals:
                return tier
        return 'common'
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        colors = {
            'common': (100, 255, 100),
            'interesting': (255, 200, 0),
            'rare': (0, 100, 255)
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
        
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        
        return np.mean(self.fps_history)
    
    def add_stats_overlay(self, frame: np.ndarray, fps: float, 
                         detection_count: int) -> np.ndarray:
        """Add professional stats overlay to frame"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 110), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        stats = [
            f"FPS: {fps:.1f}",
            f"Detections: {detection_count}",
            f"Backend: {self.backend.upper()}"
        ]
        
        y_offset = 35
        for stat in stats:
            cv2.putText(frame, stat, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        return frame
    
    def get_backend_info(self) -> Dict:
        """Get information about current backend"""
        return {
            'backend': self.backend,
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'confidence_threshold': self.confidence
        }


# Test script
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test optimized detector')
    parser.add_argument('--model', default='yolov8n.pt', help='Model path')
    parser.add_argument('--backend', default='auto', 
                       choices=['auto', 'yolov8', 'onnx', 'onnx-int8', 'tensorrt'])
    parser.add_argument('--source', default='0', help='Video source')
    parser.add_argument('--confidence', type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = OptimizedWildlifeDetector(
        args.model, 
        backend=args.backend,
        confidence=args.confidence
    )
    
    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    print(f"\n🎥 Testing {detector.backend.upper()} backend")
    print("Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated, detections, fps = detector.detect(frame)
        annotated = detector.add_stats_overlay(annotated, fps, len(detections))
        
        cv2.imshow('Optimized Wildlife Detector', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    info = detector.get_backend_info()
    print(f"\n📊 Average FPS: {info['avg_fps']:.1f}")
    print(f"🔧 Backend: {info['backend'].upper()}")
from detector import WildlifeDetector
from config import get_config
import cv2

# Use config instead of hardcoded values
config = get_config('laptop_demo')
detector = WildlifeDetector(
    model_size=config.detection.model_size,
    confidence=config.detection.confidence_threshold
)

cap = cv2.VideoCapture(config.video.source)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    annotated, detections, fps = detector.detect(frame)
    
    if config.display.show_stats_overlay:
        annotated = detector.add_stats_overlay(annotated, fps, len(detections))
    
    cv2.imshow(config.display.window_name, annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
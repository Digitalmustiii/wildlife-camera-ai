"""
Wildlife Camera AI - Complete Integrated System
Detection + Database + Alerts + Telegram + Web API
"""

import cv2
import argparse
from pathlib import Path
from datetime import datetime
import uuid
import threading
import asyncio

from detector_optimized import OptimizedWildlifeDetector
from config import get_config, Config
from database import WildlifeDatabase
from alert_manager import AlertManager, console_alert_handler, file_alert_handler
from telegram_bot import create_telegram_alert_handler
from api_server import create_api


class WildlifeCamera:
    """Complete wildlife monitoring system with web dashboard"""
    
    def __init__(self, config: Config, enable_api: bool = True, backend: str = 'auto'):
        self.config = config
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.enable_api = enable_api
        self.backend = backend
        
        print("🔧 Initializing Wildlife Camera System...")
        
        # Build model path from model size
        model_path = f'yolov8{config.detection.model_size}.pt'
        
        self.detector = OptimizedWildlifeDetector(
            model_path=model_path,
            backend=backend,
            confidence=config.detection.confidence_threshold
        )
        print("✅ Detector loaded")
        
        self.database = WildlifeDatabase(str(config.recording.database_path))
        print("✅ Database connected")
        
        self.alert_manager = AlertManager(config.alert)
        self._setup_alert_handlers()
        print("✅ Alert system ready")
        
        # API setup
        if self.enable_api:
            self.app, self.api_instance = create_api(config, self.database)
            self.api_thread = None
            print("✅ API server configured")
        
        self.frame_count = 0
        self.detection_count = 0
        self.species_seen = set()
        self.running = False
        
    def _setup_alert_handlers(self):
        """Configure alert notification handlers"""
        self.alert_manager.register_handler(console_alert_handler)
        self.alert_manager.register_handler(file_alert_handler())
        
        if self.config.alert.telegram_enabled:
            if self.config.alert.telegram_bot_token and self.config.alert.telegram_chat_id:
                try:
                    handler = create_telegram_alert_handler(
                        self.config.alert.telegram_bot_token,
                        self.config.alert.telegram_chat_id
                    )
                    self.alert_manager.register_handler(handler)
                    print("✅ Telegram notifications enabled")
                except Exception as e:
                    print(f"⚠️  Telegram setup failed: {e}")
    
    def start_api_server(self):
        """Start FastAPI server in background thread"""
        if not self.enable_api:
            return
        
        def run_server():
            import uvicorn
            uvicorn.run(self.app, host="0.0.0.0", port=8000, log_level="warning")
        
        self.api_thread = threading.Thread(target=run_server, daemon=True)
        self.api_thread.start()
        print("✅ API server started on http://localhost:8000")
        print("📊 Dashboard: Open dashboard.html in your browser")
    
    def run(self):
        """Start camera processing loop"""
        if self.enable_api:
            self.start_api_server()
        
        cap = cv2.VideoCapture(self.config.video.source)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.config.video.source}")
        
        if self.config.video.is_webcam:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.video.display_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.video.display_height)
        
        print("\n" + "="*60)
        print("🎥 WILDLIFE CAMERA STARTED")
        print("="*60)
        print(f"📹 Source: {'Webcam' if self.config.video.is_webcam else self.config.video.source}")
        print(f"🤖 Model: YOLOv8{self.config.detection.model_size}")
        print(f"🚀 Backend: {self.backend.upper()}")
        print(f"⚡ Confidence: {self.config.detection.confidence_threshold}")
        print(f"💾 Database: {self.config.recording.database_path}")
        print(f"🔔 Alerts: {'Enabled' if self.config.alert.enabled else 'Disabled'}")
        print(f"📱 Telegram: {'Enabled' if self.config.alert.telegram_enabled else 'Disabled'}")
        if self.enable_api:
            print(f"🌐 API: http://localhost:8000")
            print(f"📊 Dashboard: Open dashboard.html")
        print("="*60)
        print("\nControls: 'q'=quit | 's'=snapshot | 'p'=pause | 'd'=stats\n")
        
        self.running = True
        paused = False
        
        try:
            while self.running:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        if self.config.video.is_file:
                            print("\n✅ Video processing complete")
                            break
                        continue
                    
                    self.frame_count += 1
                    
                    annotated, detections, fps = self.detector.detect(frame)
                    
                    # Update API with current frame
                    if self.enable_api and detections:
                        self.api_instance.update_frame(annotated, [
                            {
                                'species': d['species'],
                                'confidence': d['confidence'],
                                'rarity': d['rarity'],
                                'bbox': d['bbox']
                            } for d in detections
                        ])
                    
                    for detection in detections:
                        self._process_detection(detection, frame)
                    
                    if self.config.display.show_stats_overlay:
                        annotated = self.detector.add_stats_overlay(
                            annotated, fps, len(detections)
                        )
                        annotated = self._add_session_stats(annotated)
                    
                    current_frame = annotated
                
                cv2.imshow(self.config.display.window_name, current_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('s'):
                    self._save_snapshot(current_frame)
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'⏸️  PAUSED' if paused else '▶️  RESUMED'}")
                elif key == ord('d'):
                    self._print_live_stats()
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._print_final_summary()
    
    def _process_detection(self, detection: dict, frame):
        """Process single detection: log to DB, check alerts, save snapshot"""
        snapshot_path = None
        
        if self.config.recording.save_snapshots:
            snapshot_path = self._auto_save_snapshot(detection, frame)
        
        self.database.log_detection(
            detection,
            snapshot_path=snapshot_path,
            session_id=self.session_id
        )
        
        self.detection_count += 1
        self.species_seen.add(detection['species'])
        
        self.alert_manager.process_detection(detection, snapshot_path)
    
    def _auto_save_snapshot(self, detection: dict, frame) -> str:
        """Automatically save snapshot for detection"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        species = detection['species']
        
        species_dir = self.config.recording.snapshots_dir / species
        species_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{species}_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
        filepath = species_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        return str(filepath)
    
    def _save_snapshot(self, frame):
        """Manual snapshot save (s key)"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.config.recording.snapshots_dir / f"manual_{timestamp}.jpg"
        cv2.imwrite(str(filepath), frame)
        print(f"📸 Snapshot saved: {filepath}")
    
    def _add_session_stats(self, frame):
        """Add session statistics overlay"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 280, 10), (w - 10, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        stats = [
            f"Session: {self.session_id[-6:]}",
            f"Frames: {self.frame_count}",
            f"Detections: {self.detection_count}",
            f"Species: {len(self.species_seen)}",
        ]
        
        y_offset = 35
        for stat in stats:
            cv2.putText(frame, stat, (w - 265, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        return frame
    
    def _print_live_stats(self):
        """Print live statistics (d key)"""
        print("\n" + "="*50)
        print("📊 LIVE STATISTICS")
        print("="*50)
        print(f"Frames processed: {self.frame_count}")
        print(f"Total detections: {self.detection_count}")
        print(f"Unique species: {len(self.species_seen)}")
        print(f"Species seen: {', '.join(sorted(self.species_seen))}")
        
        alert_summary = self.alert_manager.get_session_summary()
        print(f"\nAlerts sent: {alert_summary['total_alerts']}")
        
        cooldowns = self.alert_manager.get_cooldown_status()
        if cooldowns:
            print("\nCooldowns active:")
            for species, remaining in cooldowns.items():
                print(f"  {species}: {remaining:.0f}s")
        
        print("="*50 + "\n")
    
    def _print_final_summary(self):
        """Print final session summary"""
        print("\n" + "="*60)
        print("📊 SESSION SUMMARY")
        print("="*60)
        print(f"Session ID: {self.session_id}")
        print(f"Total frames: {self.frame_count}")
        print(f"Total detections: {self.detection_count}")
        print(f"Unique species: {len(self.species_seen)}")
        
        db_stats = self.database.get_total_stats()
        print(f"\nDatabase totals:")
        print(f"  All-time detections: {db_stats.get('total_detections', 0)}")
        print(f"  All-time species: {db_stats.get('unique_species', 0)}")
        
        recent = self.database.get_species_stats()
        if recent:
            print(f"\nTop species (all time):")
            for species in recent[:5]:
                print(f"  {species['species']}: {species['total_sightings']} sightings")
        
        alert_summary = self.alert_manager.get_session_summary()
        print(f"\nAlerts this session: {alert_summary['total_alerts']}")
        if alert_summary['species_breakdown']:
            print("  Breakdown:")
            for species, count in alert_summary['species_breakdown'].items():
                print(f"    {species}: {count}")
        
        print("="*60)


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Wildlife Camera AI - Complete monitoring system with web dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--backend', type=str, default='auto',
                       choices=['auto', 'yolov8', 'onnx', 'onnx-int8', 'tensorrt'],
                       help='Inference backend (auto=best available)')
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--config', type=str)
    parser.add_argument('--telegram-token', type=str)
    parser.add_argument('--telegram-chat', type=str)
    parser.add_argument('--no-alerts', action='store_true')
    parser.add_argument('--no-api', action='store_true', help='Disable web API/dashboard')
    parser.add_argument('--save-config', type=str)
    
    args = parser.parse_args()
    
    if args.config:
        config = Config(args.config)
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        preset = 'video_file' if isinstance(source, str) and Path(source).exists() else 'laptop_demo'
        config = get_config(preset)
        
        config.video.source = source
        config.detection.model_size = args.model
        config.detection.confidence_threshold = args.confidence
    
    if args.no_alerts:
        config.alert.enabled = False
    
    if args.telegram_token:
        config.alert.telegram_enabled = True
        config.alert.telegram_bot_token = args.telegram_token
        config.alert.telegram_chat_id = args.telegram_chat
    
    if args.save_config:
        config.save(args.save_config)
        print(f"✅ Config saved to {args.save_config}")
        return
    
    config.detection.validate()
    config.recording.create_directories()
    
    camera = WildlifeCamera(config, enable_api=not args.no_api, backend=args.backend)
    camera.run()


if __name__ == '__main__':
    main()
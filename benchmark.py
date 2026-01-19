"""
Performance Benchmark Tool
Compare YOLOv8, ONNX, and TensorRT backends
"""

import cv2
import numpy as np
import time
from pathlib import Path
from detector_optimized import OptimizedWildlifeDetector
import json


class BenchmarkRunner:
    """Benchmark different inference backends"""
    
    def __init__(self, model_path: str = 'yolov8n.pt'):
        self.model_path = model_path
        self.results = {}
        
    def benchmark_backend(self, backend: str, num_frames: int = 100, 
                         video_source: str = '0') -> dict:
        """
        Benchmark a specific backend
        
        Args:
            backend: Backend to test
            num_frames: Number of frames to process
            video_source: Video source (0 for webcam, or video file)
            
        Returns:
            Benchmark results dict
        """
        print(f"\n{'='*60}")
        print(f"🔬 Benchmarking: {backend.upper()}")
        print(f"{'='*60}")
        
        try:
            # Initialize detector
            detector = OptimizedWildlifeDetector(
                self.model_path,
                backend=backend,
                confidence=0.5
            )
            
            # Open video source
            source = int(video_source) if video_source.isdigit() else video_source
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {video_source}")
            
            # Warmup (10 frames)
            print("🔥 Warming up...")
            for _ in range(10):
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                detector.detect(frame)
            
            # Benchmark
            print(f"⏱️  Processing {num_frames} frames...")
            
            frame_times = []
            detection_counts = []
            
            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    # Loop video if file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                
                start = time.time()
                _, detections, _ = detector.detect(frame)
                elapsed = time.time() - start
                
                frame_times.append(elapsed)
                detection_counts.append(len(detections))
                
                if (i + 1) % 20 == 0:
                    avg_fps = 1.0 / np.mean(frame_times[-20:])
                    print(f"  Progress: {i+1}/{num_frames} frames | FPS: {avg_fps:.1f}")
            
            cap.release()
            
            # Calculate statistics
            frame_times = np.array(frame_times)
            avg_time = np.mean(frame_times) * 1000  # ms
            std_time = np.std(frame_times) * 1000
            min_time = np.min(frame_times) * 1000
            max_time = np.max(frame_times) * 1000
            avg_fps = 1.0 / np.mean(frame_times)
            
            results = {
                'backend': backend,
                'avg_inference_time_ms': avg_time,
                'std_inference_time_ms': std_time,
                'min_inference_time_ms': min_time,
                'max_inference_time_ms': max_time,
                'avg_fps': avg_fps,
                'total_frames': num_frames,
                'avg_detections': np.mean(detection_counts)
            }
            
            print(f"\n✅ Benchmark complete!")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Avg Inference: {avg_time:.2f}ms (±{std_time:.2f}ms)")
            
            return results
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return None
    
    def compare_all(self, video_source: str = '0', num_frames: int = 100):
        """Compare all available backends"""
        print("\n" + "="*60)
        print("🏁 WILDLIFE CAMERA - PERFORMANCE BENCHMARK")
        print("="*60)
        
        # Detect available backends
        backends_to_test = ['yolov8']
        
        base_name = Path(self.model_path).stem
        if Path(f"{base_name}.onnx").exists():
            backends_to_test.append('onnx')
        if Path(f"{base_name}_int8.onnx").exists():
            backends_to_test.append('onnx-int8')
        if Path(f"{base_name}.engine").exists():
            backends_to_test.append('tensorrt')
        
        print(f"\n📋 Testing backends: {', '.join(backends_to_test)}")
        print(f"📊 Frames per test: {num_frames}")
        
        # Run benchmarks
        results = {}
        for backend in backends_to_test:
            result = self.benchmark_backend(backend, num_frames, video_source)
            if result:
                results[backend] = result
        
        # Print comparison
        self._print_comparison(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _print_comparison(self, results: dict):
        """Print comparison table"""
        if not results:
            print("\n❌ No results to compare")
            return
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE COMPARISON")
        print("="*60)
        
        # Table header
        print(f"\n{'Backend':<15} {'FPS':<10} {'Inference (ms)':<18} {'Speedup':<10}")
        print("-" * 60)
        
        # Baseline (YOLOv8)
        baseline_fps = results.get('yolov8', {}).get('avg_fps', 0)
        
        # Sort by FPS
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1]['avg_fps'], 
                              reverse=True)
        
        for backend, result in sorted_results:
            fps = result['avg_fps']
            inf_time = result['avg_inference_time_ms']
            std_time = result['std_inference_time_ms']
            speedup = fps / baseline_fps if baseline_fps > 0 else 1.0
            
            speedup_str = f"{speedup:.2f}x" if backend != 'yolov8' else "baseline"
            
            print(f"{backend.upper():<15} {fps:>6.1f}     "
                  f"{inf_time:>6.1f} ±{std_time:>4.1f}    {speedup_str:<10}")
        
        print("-" * 60)
        
        # Winner
        best_backend = sorted_results[0][0]
        best_fps = sorted_results[0][1]['avg_fps']
        
        print(f"\n🏆 Winner: {best_backend.upper()} with {best_fps:.1f} FPS")
        
        if baseline_fps > 0:
            total_speedup = best_fps / baseline_fps
            print(f"⚡ Total speedup: {total_speedup:.2f}x over baseline")
    
    def _save_results(self, results: dict):
        """Save benchmark results to JSON"""
        output_file = 'benchmark_results.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Benchmark wildlife detection backends',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark (100 frames, webcam)
  python benchmark.py
  
  # Thorough benchmark (500 frames)
  python benchmark.py --frames 500
  
  # Benchmark with video file
  python benchmark.py --source wildlife.mp4 --frames 200
  
  # Benchmark specific backend only
  python benchmark.py --backend onnx-int8
        """
    )
    
    parser.add_argument('--model', default='yolov8n.pt',
                       help='Model path')
    parser.add_argument('--source', default='0',
                       help='Video source (0 for webcam or video file)')
    parser.add_argument('--frames', type=int, default=100,
                       help='Number of frames to benchmark')
    parser.add_argument('--backend', type=str,
                       help='Test specific backend only (yolov8, onnx, onnx-int8, tensorrt)')
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.model)
    
    if args.backend:
        # Single backend benchmark
        result = runner.benchmark_backend(args.backend, args.frames, args.source)
        if result:
            print(f"\n✅ Benchmark complete!")
    else:
        # Compare all backends
        runner.compare_all(args.source, args.frames)


if __name__ == '__main__':
    main()
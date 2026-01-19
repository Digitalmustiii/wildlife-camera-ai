"""
Model Optimization Pipeline
Convert YOLOv8 to ONNX and TensorRT for faster inference
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import onnx
import os


class ModelOptimizer:
    """Optimize YOLOv8 models for edge deployment"""
    
    def __init__(self, model_path: str = 'yolov8n.pt'):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.base_name = Path(model_path).stem
        
    def export_to_onnx(self, simplify: bool = True, dynamic: bool = False) -> str:
        """
        Export YOLOv8 to ONNX format
        
        Args:
            simplify: Apply ONNX graph simplification
            dynamic: Use dynamic input shapes (slower but flexible)
            
        Returns:
            Path to exported ONNX model
        """
        print(f"📦 Exporting {self.model_path} to ONNX...")
        
        onnx_path = self.model.export(
            format='onnx',
            simplify=simplify,
            dynamic=dynamic,
            imgsz=640
        )
        
        print(f"✅ ONNX model saved: {onnx_path}")
        
        # Verify model
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)
        print("✅ ONNX model verified")
        
        return onnx_path
    
    def quantize_onnx(self, onnx_path: str, quantization_type: str = 'int8') -> str:
        """
        Quantize ONNX model for faster inference
        
        Args:
            onnx_path: Path to ONNX model
            quantization_type: 'int8' or 'fp16'
            
        Returns:
            Path to quantized model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            print(f"🔧 Quantizing ONNX model to {quantization_type.upper()}...")
            
            if quantization_type == 'int8':
                output_path = onnx_path.replace('.onnx', '_int8.onnx')
                quantize_dynamic(
                    onnx_path,
                    output_path,
                    weight_type=QuantType.QUInt8
                )
            else:
                # FP16 quantization
                try:
                    from onnxmltools.utils.float16_converter import convert_float_to_float16
                    model = onnx.load(onnx_path)
                    model_fp16 = convert_float_to_float16(model)
                    output_path = onnx_path.replace('.onnx', '_fp16.onnx')
                    onnx.save(model_fp16, output_path)
                except ImportError:
                    print("⚠️  FP16 conversion requires: pip install onnxmltools")
                    return onnx_path
            
            print(f"✅ Quantized model saved: {output_path}")
            
            # Compare sizes
            original_size = os.path.getsize(onnx_path) / (1024**2)
            quantized_size = os.path.getsize(output_path) / (1024**2)
            reduction = ((original_size - quantized_size) / original_size) * 100
            
            print(f"📉 Size reduction: {original_size:.2f}MB → {quantized_size:.2f}MB ({reduction:.1f}% smaller)")
            
            return output_path
            
        except ImportError:
            print("⚠️  ONNX quantization requires: pip install onnxruntime")
            return onnx_path
    
    def export_to_tensorrt(self, device: int = 0, fp16: bool = True) -> str:
        """
        Export to TensorRT (NVIDIA GPUs/Jetson only)
        
        Args:
            device: GPU device index
            fp16: Use FP16 precision (2x faster, minimal accuracy loss)
            
        Returns:
            Path to TensorRT engine
        """
        print(f"🚀 Exporting to TensorRT (FP16={fp16})...")
        
        try:
            trt_path = self.model.export(
                format='engine',
                device=device,
                half=fp16,
                imgsz=640
            )
            
            print(f"✅ TensorRT engine saved: {trt_path}")
            return trt_path
            
        except Exception as e:
            print(f"⚠️  TensorRT export failed: {e}")
            print("💡 TensorRT requires NVIDIA GPU and TensorRT installed")
            print("   Install: pip install nvidia-tensorrt")
            return None
    
    def optimize_all(self):
        """Run complete optimization pipeline"""
        results = {
            'original': self.model_path,
            'onnx': None,
            'onnx_int8': None,
            'tensorrt': None
        }
        
        print("\n" + "="*60)
        print("🔥 WILDLIFE CAMERA - MODEL OPTIMIZATION")
        print("="*60 + "\n")
        
        # Step 1: Export to ONNX
        try:
            onnx_path = self.export_to_onnx()
            results['onnx'] = onnx_path
            print()
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
        
        # Step 2: Quantize ONNX
        if results['onnx']:
            try:
                int8_path = self.quantize_onnx(results['onnx'], 'int8')
                results['onnx_int8'] = int8_path
                print()
            except Exception as e:
                print(f"⚠️  Quantization skipped: {e}")
        
        # Step 3: Export to TensorRT (optional)
        try:
            trt_path = self.export_to_tensorrt()
            results['tensorrt'] = trt_path
            print()
        except Exception as e:
            print(f"⚠️  TensorRT export skipped (optional): {e}")
        
        # Summary
        print("="*60)
        print("📊 OPTIMIZATION SUMMARY")
        print("="*60)
        for model_type, path in results.items():
            if path:
                size = os.path.getsize(path) / (1024**2)
                print(f"✅ {model_type.upper():15} {path} ({size:.2f} MB)")
            else:
                print(f"⏸️  {model_type.upper():15} Not created")
        print("="*60 + "\n")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Optimize YOLOv8 models for edge deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize default model (yolov8n.pt)
  python optimize_model.py
  
  # Optimize specific model
  python optimize_model.py --model yolov8s.pt
  
  # ONNX only (skip TensorRT)
  python optimize_model.py --onnx-only
  
  # TensorRT with FP32 (more accurate, slower)
  python optimize_model.py --no-fp16
        """
    )
    
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLOv8 model to optimize')
    parser.add_argument('--onnx-only', action='store_true',
                       help='Only export ONNX (skip TensorRT)')
    parser.add_argument('--no-fp16', action='store_true',
                       help='Use FP32 for TensorRT (slower but more accurate)')
    parser.add_argument('--skip-quantization', action='store_true',
                       help='Skip INT8 quantization')
    
    args = parser.parse_args()
    
    # Check if model exists, download if not
    model_path = args.model
    if not Path(model_path).exists():
        print(f"📥 Downloading {model_path}...")
        YOLO(model_path)  # Auto-downloads
    
    # Optimize
    optimizer = ModelOptimizer(model_path)
    
    if args.onnx_only:
        onnx_path = optimizer.export_to_onnx()
        if not args.skip_quantization:
            optimizer.quantize_onnx(onnx_path)
    else:
        optimizer.optimize_all()
    
    print("\n✅ Optimization complete!")
    print("\n💡 Usage:")
    print("  python main.py --backend onnx      # Use ONNX Runtime")
    print("  python main.py --backend onnx-int8 # Use INT8 quantized")
    print("  python main.py --backend tensorrt  # Use TensorRT (GPU)")


if __name__ == '__main__':
    main()
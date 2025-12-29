"""
Script để build TensorRT engine từ ONNX file trên Jetson Nano
Sử dụng TensorRT Python API
"""

import tensorrt as trt
import sys
import os

def build_engine_from_onnx(onnx_path, engine_path, fp16=True, workspace_size=4096):
    """
    Build TensorRT engine từ ONNX file
    
    Args:
        onnx_path: Đường dẫn đến file ONNX
        engine_path: Đường dẫn để lưu file engine
        fp16: Sử dụng FP16 precision (khuyến nghị cho Jetson Nano)
        workspace_size: Kích thước workspace (MB)
    """
    print(f"Building TensorRT engine from: {onnx_path}")
    print(f"Output: {engine_path}")
    print(f"FP16: {fp16}, Workspace: {workspace_size}MB")
    
    # Kiểm tra file ONNX tồn tại
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found: {onnx_path}")
        return False
    
    # Tạo TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Tạo builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Tạo ONNX parser
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    print("Parsing ONNX model...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Error: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    print("ONNX model parsed successfully!")
    
    # Tạo builder config
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size << 20  # Convert MB to bytes
    
    # Enable FP16 nếu được yêu cầu
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 precision enabled")
        else:
            print("Warning: FP16 not supported on this platform, using FP32")
    
    # Build engine
    print("Building engine... (This may take a few minutes)")
    try:
        engine = builder.build_engine(network, config)
        if engine is None:
            print("Error: Failed to build engine")
            return False
        
        # Serialize và lưu engine
        print("Saving engine...")
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"Engine built successfully: {engine_path}")
        return True
        
    except Exception as e:
        print(f"Error building engine: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python build_engine.py <input.onnx> <output.engine> [--fp32] [--workspace SIZE_MB]")
        print("Example: python build_engine.py ../models/yolov5n.onnx ../models/yolov5n.engine")
        print("Example: python build_engine.py ../models/yolov5n.onnx ../models/yolov5n.engine --fp32 --workspace 2048")
        sys.exit(1)
    
    onnx_path = sys.argv[1]
    engine_path = sys.argv[2]
    
    # Parse options
    fp16 = True
    workspace_size = 4096
    
    for i, arg in enumerate(sys.argv[3:], 3):
        if arg == '--fp32':
            fp16 = False
        elif arg == '--workspace' and i + 1 < len(sys.argv):
            workspace_size = int(sys.argv[i + 1])
    
    success = build_engine_from_onnx(onnx_path, engine_path, fp16, workspace_size)
    sys.exit(0 if success else 1)


"""
Script để convert YOLOv5 PyTorch model (.pt) sang file .wts
File .wts này sẽ được sử dụng để build TensorRT engine trên Jetson Nano
"""

import torch
import struct
import sys

def convert_pt_to_wts(pt_path, wts_path):
    """
    Convert PyTorch model (.pt) to .wts format
    
    Args:
        pt_path: Đường dẫn đến file .pt
        wts_path: Đường dẫn để lưu file .wts
    """
    print(f"Loading PyTorch model from: {pt_path}")
    
    # Load model
    model = torch.load(pt_path, map_location='cpu', weights_only=False)
    
    # Nếu model là dictionary (checkpoint), lấy 'model' key
    if isinstance(model, dict):
        if 'model' in model:
            model = model['model']
        elif 'state_dict' in model:
            # Nếu chỉ có state_dict, cần load lại với YOLOv5
            print("Warning: Only state_dict found. Please use full model export.")
            sys.exit(1)
    
    # Lấy state_dict
    if hasattr(model, 'state_dict'):
        state_dict = model.state_dict()
    elif isinstance(model, dict):
        state_dict = model
    else:
        print("Error: Cannot extract state_dict from model")
        sys.exit(1)
    
    # Ghi vào file .wts
    print(f"Writing weights to: {wts_path}")
    with open(wts_path, 'w') as f:
        f.write('{}\n'.format(len(state_dict)))
        
        for k, v in state_dict.items():
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')
    
    print("Conversion completed successfully!")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python gen_wts.py <input.pt> <output.wts>")
        print("Example: python gen_wts.py ../models/yolov5n.pt ../models/yolov5n.wts")
        sys.exit(1)
    
    pt_path = sys.argv[1]
    wts_path = sys.argv[2]
    
    convert_pt_to_wts(pt_path, wts_path)


import sys
import argparse
import os
import struct
import torch
from utils.torch_utils import select_device

def parse_args():
    parser = argparse.ArgumentParser(description='Convert .pt file to .wts')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-o', '--output', required=True, help='Output (.wts) file path (required)')
    parser.add_argument('-t', '--type', type=str, default='detect', help='model type: detect/cls/seg')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid input file')
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    return args

def main():
    args = parse_args()
    pt_file = args.weights
    wts_file = args.output

    print(f'Loading {pt_file}')
    device = select_device('cpu')
    
    # Load model
    model = torch.load(pt_path, map_location='cpu', weights_only=False)
    
    # Load weights properly
    if model.get('model'):
        model = model['model']
    
    model.to(device).float()
    model.eval()  # QUAN TRỌNG: Phải eval() để register các buffer như strides/anchors

    print(f'Writing into {wts_file}')
    with open(wts_file, 'w') as f:
        # Ghi số lượng phần tử
        f.write('{}\n'.format(len(model.state_dict().keys())))
        
        # Duyệt qua từng layer
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')

if __name__ == '__main__':
    main()

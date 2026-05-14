#!/usr/bin/env python3
"""
Unified PyTorch benchmark for all paper models.

Timing methodology matches IQFormer original paper:
  start = time.time()
  full_test_epoch(test_loader, model, device)   # DataLoader + forward + result collection
  end = time.time()
  avg_time = (end - start) / num_samples

All models run under PyTorch to ensure fair cross-model comparison.
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ===================== Model builders =====================

def build_iqformer(num_classes, frame_length):
    from model.iqformer_torch_model import build_iqformer_model
    return build_iqformer_model((2, frame_length), num_classes)

def build_fea_t(num_classes, frame_length):
    from model.fea_t_torch_model import build_fea_t_model
    return build_fea_t_model((2, frame_length), num_classes)

def build_mcldnn(num_classes, frame_length):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'IQFormer', 'model'))
    from MCLDNN import MCLDNN
    return MCLDNN(frame_length=frame_length, num_classes=num_classes)

def build_petcgdnn(num_classes, frame_length):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'IQFormer', 'model'))
    from PETCGDNN import PETCGDNN
    return PETCGDNN(num_classes=num_classes, frame_length=frame_length)

def build_amcnet(num_classes, frame_length):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'IQFormer', 'model'))
    from AMCNET import AMC_Net
    return AMC_Net(num_classes=num_classes, sig_len=frame_length)

def build_ulcnn(num_classes, frame_length):
    from model.ulcnn_torch_model import build_ulcnn_torch_model
    return build_ulcnn_torch_model((2, frame_length), num_classes)


MODEL_BUILDERS = {
    'IQFormer':  build_iqformer,
    'AMC-Net':   build_amcnet,
    'FEA-T':     build_fea_t,
    'MCLDNN':    build_mcldnn,
    'PETCGDNN':  build_petcgdnn,
    'ULCNN':     build_ulcnn,
}


# ===================== Test epoch (mirrors IQFormer paper) =====================

def test_epoch(data_loader, model, device, criterion):
    """Full test epoch matching IQFormer paper's test_epoch():
    DataLoader iteration + GPU transfer + forward + loss + result collection.
    """
    model.eval()
    y_pred = []
    y_true = []
    num_total = 0

    with torch.no_grad():
        for (batch_x, batch_y) in tqdm(data_loader, desc='Testing', leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            batch_out = model(batch_x)
            _ = criterion(batch_out, batch_y)       # loss computation (as in original)

            preds = batch_out.cpu().detach().numpy()
            preds = preds.argmax(1).tolist()
            trues = batch_y.cpu().detach().numpy().tolist()

            y_pred.extend(preds)
            y_true.extend(trues)
            num_total += batch_x.size(0)

    return y_true, y_pred


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser(description='Unified PyTorch benchmark')
    parser.add_argument('--batch_size', type=int, default=400,
                        help='Batch size (default: 400, matching IQFormer paper)')
    parser.add_argument('--num_samples', type=int, default=22000,
                        help='Number of test samples (default: 22000)')
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--frame_length', type=int, default=128)
    parser.add_argument('--num_warmup', type=int, default=1,
                        help='Number of warmup runs before timing')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of timed runs to average')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to benchmark (default: all)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', end='')
    if device.type == 'cuda':
        print(f' ({torch.cuda.get_device_name(0)})')
    else:
        print()
    print(f'batch_size={args.batch_size}, num_samples={args.num_samples}, '
          f'frame_length={args.frame_length}, num_runs={args.num_runs}\n')

    # Synthetic test data
    X = torch.randn(args.num_samples, 2, args.frame_length)
    y = torch.randint(0, args.num_classes, (args.num_samples,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    models_to_run = args.models if args.models else list(MODEL_BUILDERS.keys())

    results = []

    for name in models_to_run:
        if name not in MODEL_BUILDERS:
            print(f'Unknown model: {name}, skipping')
            continue

        print(f'--- {name} ---')
        try:
            model = MODEL_BUILDERS[name](args.num_classes, args.frame_length)
        except Exception as e:
            print(f'  BUILD ERROR: {e}\n')
            results.append((name, 'ERROR', 'ERROR', 'ERROR'))
            continue

        model = model.to(device)
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        print(f'  Parameters: {params:,}')

        # Warmup
        for _ in range(args.num_warmup):
            test_epoch(loader, model, device, criterion)
        torch.cuda.synchronize() if device.type == 'cuda' else None

        # Timed runs
        run_times = []
        for r in range(args.num_runs):
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t0 = time.time()
            test_epoch(loader, model, device, criterion)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t1 = time.time()
            elapsed = t1 - t0
            per_sample = elapsed / args.num_samples * 1000  # ms
            run_times.append(per_sample)
            print(f'  Run {r+1}: {elapsed:.3f}s  ({per_sample:.4f} ms/sample)')

        avg_ms = np.mean(run_times)
        std_ms = np.std(run_times)
        print(f'  Average: {avg_ms:.4f} ± {std_ms:.4f} ms/sample\n')
        results.append((name, params, avg_ms, std_ms))

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Summary table
    print('=' * 65)
    print(f'  {"Model":<12} {"Parameters":>12} {"ms/sample":>12} {"± std":>10}')
    print(f'  {"-"*12} {"-"*12} {"-"*12} {"-"*10}')
    for name, params, avg, std in results:
        if params == 'ERROR':
            print(f'  {name:<12} {"ERROR":>12} {"":>12} {"":>10}')
        else:
            print(f'  {name:<12} {params:>12,} {avg:>12.4f} {f"± {std:.4f}":>10}')
    print(f'  {"GPR Denoise":<12} {"--":>12} {"0.0125":>12} {"":>10}')
    print('=' * 65)

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'benchmark_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'pytorch_unified_benchmark.txt')
    with open(out_path, 'w') as f:
        f.write(f'Device: {device}')
        if device.type == 'cuda':
            f.write(f' ({torch.cuda.get_device_name(0)})')
        f.write(f'\nbatch_size={args.batch_size}, num_samples={args.num_samples}, '
                f'frame_length={args.frame_length}, num_runs={args.num_runs}\n\n')
        f.write(f'{"Model":<12} {"Parameters":>12} {"ms/sample":>12} {"std":>10}\n')
        f.write(f'{"-"*46}\n')
        for name, params, avg, std in results:
            if params == 'ERROR':
                f.write(f'{name:<12} {"ERROR":>12}\n')
            else:
                f.write(f'{name:<12} {params:>12,} {avg:>12.4f} {std:>10.4f}\n')
        f.write(f'{"GPR Denoise":<12} {"--":>12} {"0.0125":>12}\n')
    print(f'\nSaved to: {out_path}')


if __name__ == '__main__':
    main()

import os, pickle, numpy as np, json
root = '/home/lijunkai/radioML-v4'
input_pkl = os.path.join(root, 'data', 'RML2016.10a_dict.pkl')
output_dir = os.path.join(root, 'ULCNN', 'train_RML')
os.makedirs(output_dir, exist_ok=True)
print('Loading:', input_pkl)
with open(input_pkl, 'rb') as f:
    try:
        data = pickle.load(f)
    except Exception:
        f.seek(0)
        data = pickle.load(f, encoding='latin1')

# Collect classes and SNRs
mods = sorted({k[0] for k in data.keys()})
snrs = sorted({k[1] for k in data.keys()})
print('Found mods:', mods)
print('Found SNRs:', snrs[:5], '... total', len(snrs))

X_list = []
Y_list = []
for m in mods:
    for snr in snrs:
        k = (m, snr)
        if k not in data:
            continue
        arr = np.array(data[k])
        # normalize shape to (N, 128, 2)
        if arr.ndim != 3:
            raise RuntimeError(f'Unexpected array ndim for {k}: {arr.shape}')
        if arr.shape[1] == 2 and arr.shape[2] == 128:
            arr = np.transpose(arr, (0, 2, 1))
        elif arr.shape[1] == 128 and arr.shape[2] == 2:
            pass
        else:
            raise RuntimeError(f'Unexpected array shape for {k}: {arr.shape}')
        X_list.append(arr.astype(np.float32, copy=False))
        Y_list.append(np.full((arr.shape[0],), mods.index(m), dtype=np.int64))

X = np.concatenate(X_list, axis=0)
Y = np.concatenate(Y_list, axis=0)
print('Final shapes:', X.shape, Y.shape)

np.save(os.path.join(output_dir, 'x_r=1.npy'), X)
np.save(os.path.join(output_dir, 'y_r=1.npy'), Y)

# Save mapping for reference
with open(os.path.join(output_dir, 'classes.json'), 'w', encoding='utf-8') as f:
    json.dump({'classes': mods}, f, ensure_ascii=False, indent=2)
print('Saved to', output_dir)
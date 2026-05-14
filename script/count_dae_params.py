#!/usr/bin/env python3
"""
Count parameters of the DAE model defined in
AMR-Benchmark/RML201610a/DAE/rmlmodels/DAE.py

Usage (from repo root):
  python script/count_dae_params.py

This script loads the original DAE model file by absolute path (to avoid
import issues caused by the hyphen in 'AMR-Benchmark') and prints total,
trainable, and non-trainable parameter counts.
"""

import os
import sys
import importlib.util
import traceback
import numpy as np
import types


def load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    # Compatibility shim: old code imports CuDNNLSTM from keras.layers, which no longer exists in Keras 3.
    # We alias CuDNNLSTM to LSTM before executing the target module so its imports succeed.
    try:
        import keras.layers as _kl  # type: ignore
        if not hasattr(_kl, 'CuDNNLSTM'):
            setattr(_kl, 'CuDNNLSTM', _kl.LSTM)
    except Exception:
        # If keras is not available or import fails, let execution proceed to show an informative error later.
        pass
    # Compatibility shim: legacy code imports plot_model from keras.utils.vis_utils (removed in Keras 3).
    # We provide a small proxy module that exposes plot_model from current location when available.
    try:
        import sys as _sys
        import keras.utils as _ku  # type: ignore
        _plot = getattr(_ku, 'plot_model', None)
        if _plot is None:
            try:
                from tensorflow.keras.utils import plot_model as _plot  # type: ignore
            except Exception:
                _plot = None
        vis_mod_name = 'keras.utils.vis_utils'
        if vis_mod_name not in _sys.modules:
            _vis_mod = types.ModuleType(vis_mod_name)
            if _plot is None:
                def _noop_plot_model(*args, **kwargs):  # type: ignore
                    raise ImportError('plot_model is not available in current Keras/TensorFlow installation')
                _vis_mod.plot_model = _noop_plot_model  # type: ignore[attr-defined]
            else:
                _vis_mod.plot_model = _plot  # type: ignore[attr-defined]
            _sys.modules[vis_mod_name] = _vis_mod
    except Exception:
        pass
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dae_file = os.path.join(
        repo_root,
        'AMR-Benchmark', 'RML201610a', 'DAE', 'rmlmodels', 'DAE.py'
    )

    if not os.path.isfile(dae_file):
        print(f"[ERROR] DAE.py not found at: {dae_file}", file=sys.stderr)
        sys.exit(1)

    try:
        dae_mod = load_module_from_path('dae_module', dae_file)
    except Exception as e:
        print('[ERROR] Failed to import DAE module:', file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)

    if not hasattr(dae_mod, 'DAE'):
        print('[ERROR] DAE function not found in module.', file=sys.stderr)
        sys.exit(3)

    try:
        # Use default settings from the original file
        model = dae_mod.DAE(weights=None, input_shape=[128, 2], classes=11)
    except Exception:
        print('[ERROR] Failed to build DAE model:', file=sys.stderr)
        traceback.print_exc()
        sys.exit(4)

    # Count parameters
    try:
        total_params = model.count_params()
        trainable_params = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
        non_trainable_params = int(np.sum([np.prod(w.shape) for w in model.non_trainable_weights]))
    except Exception:
        print('[ERROR] Failed to count parameters:', file=sys.stderr)
        traceback.print_exc()
        sys.exit(5)

    # Pretty print
    def fmt(n: int) -> str:
        return f"{n:,}"

    print('DAE model parameter counts (from AMR-Benchmark/RML201610a/DAE/rmlmodels/DAE.py)')
    print(f"  Total params:         {fmt(total_params)}")
    print(f"  Trainable params:     {fmt(trainable_params)}")
    print(f"  Non-trainable params: {fmt(non_trainable_params)}")


if __name__ == '__main__':
    main()

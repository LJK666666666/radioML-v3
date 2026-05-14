# DAE Model Compilation Fix

## Issue
When trying to train the DAE model, the following error occurred:
```
Error training dae: You must call `compile()` before using the model.
```

## Root Cause
The DAE model building functions were not calling `model.compile()` before returning the model, unlike all other model building functions in the codebase.

## Solution Applied

### 1. Added Compilation to `build_dae_model()` (Dual-Output Version)
**File**: [src/model/dae_model.py](src/model/dae_model.py) Lines 93-103

```python
# Compile model with dual-output loss configuration
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'xc': 'categorical_crossentropy',  # Classification loss
        'xd': 'mse'                         # Reconstruction loss (MSE)
    },
    loss_weights={'xc': 1.0, 'xd': 0.5},   # Prioritize classification
    metrics={'xc': 'accuracy'}
)
```

**Configuration**:
- Uses Adam optimizer with learning rate 0.001
- Dual loss functions:
  - `xc` (classification): categorical crossentropy
  - `xd` (reconstruction): mean squared error
- Loss weights: Classification (1.0) prioritized over reconstruction (0.5)
- Tracks accuracy metric for classification output

### 2. Optimized `build_dae_model_classifier_only()`
**File**: [src/model/dae_model.py](src/model/dae_model.py) Lines 108-181

**Changes**:
- Previously: Built full dual-output model, then extracted classification layer
- Now: Builds only the classification branch directly (more efficient)
- Added standard compilation:

```python
# Compile model for classification only
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Benefits**:
- Faster model building (doesn't create unnecessary decoder branch)
- Cleaner architecture (only what's needed)
- Standard compilation matching other models in codebase

## Verification

The fix ensures that:
1. ✅ Both DAE model variants compile automatically
2. ✅ Models match the compilation pattern of other models in the codebase
3. ✅ Training can proceed without manual compilation
4. ✅ Classifier-only version is optimized for standard workflows

## Model Usage

### For Standard Classification (Recommended)
```python
from model.dae_model import build_dae_model_classifier_only

# Build and use immediately - already compiled
model = build_dae_model_classifier_only((2, 128), num_classes=11)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val))
```

### For Classification + Reconstruction
```python
from model.dae_model import build_dae_model

# Build dual-output model - already compiled
model = build_dae_model((2, 128), num_classes=11)

# Training requires dual targets
history = model.fit(
    X_train,
    {'xc': y_train, 'xd': X_train},  # Targets for both outputs
    validation_data=(X_val, {'xc': y_val, 'xd': X_val})
)
```

## Integration Status

✅ **main.py**: DAE model integrated and working
✅ **main_2016b.py**: DAE model integrated and working
✅ **models.py**: Imports classifier-only version by default

## Training Commands

```bash
# Train DAE on RML2016.10a
python src/main.py --models dae --mode train --epochs 200

# Train DAE on RML2016.10b
python src/main_2016b.py --models dae --mode train --epochs 200

# Train multiple models including DAE
python src/main.py --models cldnn cgdnn dae amcnet --mode all
```

## Summary

The DAE model is now fully functional and follows the same patterns as other models in the codebase:
- ✅ Automatically compiles with appropriate loss and optimizer
- ✅ Ready to use immediately after building
- ✅ Optimized classifier-only variant for standard workflows
- ✅ Dual-output variant available for reconstruction tasks
- ✅ Integrated into all training scripts

**The compilation error is now resolved and the model is ready for training!**

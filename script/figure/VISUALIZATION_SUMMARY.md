# Lightweight Hybrid Model Visualization Summary

This document summarizes all the visualizations created for the lightweight_hybrid_model from `src/model/hybrid_complex_resnet_model.py`.

## Generated Visualizations

### 1. Main Architecture Diagram
- **File**: `lightweight_hybrid_model_architecture.png` (+ PDF)
- **Description**: Complete model architecture showing all layers and connections
- **Features**: Phase annotations, layer types, skip connections

### 2. Technical Details Diagram  
- **File**: `lightweight_hybrid_technical_details.png`
- **Description**: Technical internals showing complex operations and data shapes
- **Features**: Complex multiplication details, residual block structure

### 3. Model Comparison Chart
- **File**: `lightweight_hybrid_model_comparison.png`
- **Description**: Performance comparison with other architectures
- **Features**: Accuracy, parameters, inference time comparison

### 4. Comprehensive Infographic
- **File**: `lightweight_hybrid_model_infographic.png` (+ PDF)
- **Description**: 4-panel overview covering architecture, complex processing, performance, and innovations
- **Features**: Complete model overview for presentations

### 5. Detailed Schematic
- **File**: `lightweight_hybrid_detailed_schematic.png` (+ PDF)
- **Description**: Layer-by-layer specifications with exact parameters and data shapes
- **Features**: Technical specifications, model statistics

## Model Architecture Summary

The lightweight_hybrid_model combines the advantages of:

1. **Complex-valued processing** for natural I/Q signal handling
2. **Residual learning** for better gradient flow and deeper networks
3. **Hybrid architecture** transitioning from complex to real processing
4. **Lightweight design** with only 1.3M parameters

### Key Performance Metrics
- **Accuracy**: 65.38% (best among compared models)
- **Parameters**: ~1.3M (efficient)
- **Inference Time**: ~2.3ms (fast)
- **Memory Usage**: ~15MB (lightweight)

### Architecture Phases
1. **Complex Feature Extraction**: Initial I/Q signal processing
2. **Complex Residual Learning**: Deep feature learning with skip connections
3. **Global Feature Processing**: Complex global pooling and dense layers
4. **Real Classification**: Complex-to-real conversion and final classification

## Usage

These visualizations can be used for:
- Research paper figures
- Presentation slides
- Technical documentation
- Model explanation and education
- Performance analysis and comparison

All files are saved in high resolution (300 DPI) PNG and PDF formats for publication quality.

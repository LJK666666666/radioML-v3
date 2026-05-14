# RadioML

This repository contains machine learning models for radio modulation classification.

## Usage

```bash

# preparation
pip install -r requirements
cd src

# evaluate
python main.py --models lightweight_hybrid --denoising_method efficient_gpr_per_sample --augment_data --mode evaluate
python main.py --models pet --denoising_method efficient_gpr_per_sample --augment_data --mode evaluate
python main.py --models mcldnn --denoising_method efficient_gpr_per_sample --augment_data --mode evaluate

# train and evaluate
python main.py --models lightweight_hybrid --epochs 200 --denoising_method efficient_gpr_per_sample --batch_size 256 --augment_data
python main.py --models micro_lightweight_hybrid --epochs 200 --denoising_method efficient_gpr_per_sample --batch_size 256 --augment_data
python main.py --models ulcnn --epochs 200 --denoising_method efficient_gpr_per_sample --batch_size 256 --augment_data
python main.py --models pet --epochs 200 --denoising_method efficient_gpr_per_sample --batch_size 256 --augment_data
python main.py --models mcldnn --epochs 200 --denoising_method efficient_gpr_per_sample --batch_size 256 --augment_data
python main.py --models cgdnn --epochs 200 --denoising_method efficient_gpr_per_sample --batch_size 256 --augment_data
python main.py --models mcnet --epochs 200 --denoising_method efficient_gpr_per_sample --batch_size 256 --augment_data
python main.py --models resnet --epochs 200 --denoising_method efficient_gpr_per_sample --batch_size 256 --augment_data
python main.py --models complex_nn --epochs 200 --denoising_method efficient_gpr_per_sample --batch_size 256 --augment_data

```

To run the project, use `src/main.py`. Key command-line arguments include:

*   `--mode`: Mode of operation. Choices: `explore`, `train`, `evaluate`, `all`. Default: `all`.
*   `--model_type`: Model architecture to use. Choices: .
*   `--dataset_path`: Path to the RadioML dataset. Default: `../RML2016.10a_dict.pkl`.
*   `--epochs`: Number of training epochs. Default: 400.
*   `--batch_size`: Batch size for training. Default: 128.
*   `--augment_data`: Enable data augmentation.
*   `--denoising_method`: Denoising method to apply to the input signals. Default: `gpr`.
    *   `efficient_gpr_per_sample`: Gaussian Process Regression. (Default kernel is RBF; Matern and RationalQuadratic also available).
    *   `none`: No denoising is applied.


## Project Structure

- `src/`: Source code for all models
  - Models implementations (CNN 1D, CNN 2D, Complex NN, ResNet, Transformer)
  - Utility functions and callbacks
- `model_weight_saved/`: Saved model weights (managed with Git LFS)
- `output/models/`: Output model files (managed with Git LFS)
- `RML2016.10a_dict.pkl`: The RadioML dataset (managed with Git LFS)
- `projects/`: Contains submodules of related projects
  - `ULCNN`: Implementation of the ULCNN architecture

## Getting Started

To use this code, you will need to:

1. Download the RadioML dataset (RML2016.10a) from the official website
2. Place the dataset file in `data`
3. Run the training scripts to train the models or use pre-trained weights

## Models

MCNET、CGDNN、ULCNN、MCLDNN、PETCGDNN、ComplexCNN、ResNet、Complex-ResNet-mini、Complex-ResNet。


## Results

The following table shows the performance of different neural network architectures with GPR denoising and data augmentation techniques:

| Model | Accuracy (%) | +Aug (%) | +GPR (%) | +Aug+GPR (%) |
|-------|--------------|----------|----------|--------------|
| MCNET | 56.73 | 57.28 | 58.82 | 61.78 |
| CGDNN | 53.77 | 59.46 | 58.84 | 61.08 |
| ULCNN | 56.15 | 57.26 | 58.23 | 62.28 |
| ComplexCNN | 57.11 | 58.07 | 61.78 | 62.89 |
| ResNet | 55.37 | 59.34 | 62.15 | 63.24 |
| MCLDNN | 60.27 | 61.71 | **64.65** | 65.17 |
| PETCGDNN | **60.72** | **61.98** | 63.28 | 65.26 |
| Complex-ResNet-mini | 57.06 | 58.74 | 60.58 | 64.94 |
| Complex-ResNet | 56.94 | 59.60 | 61.49 | **65.52** |

**Notes:**
- Aug: Rotation data augmentation
- GPR: Gaussian Process Regression denoising
- Bold values indicate the best performance in each column
- The best overall accuracy of **65.52%** is achieved by Complex-ResNet with both augmentation and GPR denoising



# Skin Lesion Segmentation Project

This project focuses on developing a deep learning approach for automatic segmentation of skin lesions in dermoscopic images. The goal is to accurately identify lesion regions, which is important for diagnosing skin conditions such as melanoma. The project is implemented in a Jupyter Notebook designed for clarity, ease of use, and reproducibility.

---

## Project Overview

The primary aim is to build a **U-Net-based segmentation pipeline** that uses paired images and masks to train a model to detect lesions. The method employs supervised learning and is evaluated using common medical image segmentation metrics.

---

## Dataset

- The project uses an open-source dataset containing RGB images and corresponding binary lesion masks.
- Proper splits for training, validation, and testing are applied.
- Images and masks are resized and preprocessed to be compatible with PyTorch.

---

## Key Libraries

- **PyTorch** for building and training the neural network model.
- **Albumentations** for image and mask augmentations like rotation, flipping, resizing, and normalization.
- **scikit-learn** for calculating performance metrics and analysis.
- **NumPy, PIL, matplotlib** for data handling and visualizations.

---

## Code Workflow

### 1. Data Preparation and Augmentation

- Custom Dataset classes load and preprocess data.
- Various augmentations simulate real-world image diversity and prepare fixed-size inputs.
- PyTorch DataLoaders batch and shuffle data efficiently.

### 2. Model Architecture

- A custom U-Net architecture, balancing accuracy and memory efficiency.
- Architecture includes encoder, bottleneck, and decoder blocks with skip connections to preserve spatial information.

### 3. Training Setup

- Loss functions combined Binary Cross-Entropy and Dice loss to improve segmentation.
- Adam optimizer selected for reliable and fast convergence.
- Learning rate scheduler and early stopping help optimize performance and prevent overfitting.
- Training and validation losses are tracked and visualized.

### 4. Evaluation Metrics

- Performance is measured with:
  - Dice Score (F1), Jaccard Index (IoU), Precision, Recall, Pixel Accuracy, Sensitivity, and Specificity.
- Confusion matrices and result visualizations support detailed analysis.

### 5. Output and Visualization

- Visual comparisons between predicted masks and ground truth.
- Training graphs to monitor loss and accuracy.

---

## Results

- The optimized model achieves:
  - Dice Score approx. 0.87
  - Jaccard Index approx. 0.79
  - Precision approx. 0.91
  - Recall approx. 0.87
- Inference speed is around 22 images per second, showing practical applicability.

---

## Customization and Extensions

- Model parameters such as channel sizes, image dimensions, and learning rates can be adjusted.
- The modular design allows easy replacement of augmentations, architectures, or metrics.
- Well-commented code explains each important step and decision.

---

## Getting Started

1. Set up a Python environment with required dependencies (see notebook).
2. Update dataset paths to your local image and mask folders.
3. Run notebook cells in order to train and evaluate the model.
4. Compatible with CPU or GPU for training.

---

## Acknowledgements

- Inspired by leading U-Net research and dermoscopic lesion segmentation work.
- Powered by PyTorch, Albumentations, and scikit-learn to enable fast experimentation.
- Compatible with commonly used datasets like ISIC and PH2.

---

## License

This project is intended for academic, research, and educational use. Please provide appropriate credit when using the code or models.

---

*For technical details, please refer to comments and explanations in the notebook code cells.*


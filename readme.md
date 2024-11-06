# Enhanced ADA-Net with Semi-Supervised Learning

This repository contains updates to ADA-Net incorporating semi-supervised learning techniques aimed at improving model performance. Key changes involve an adaptive sample selection strategy and CNN optimization for improved efficiency and accuracy. To explore the summary of changes performed and results achieved, open the `POC_Adanet.pdf` file in the repo.

## Updates Overview

1. **Adaptive Sample Selection**:
   - Implemented an adaptive selection strategy to improve distribution alignment by focusing on low-confidence, pseudo-labeled samples near decision boundaries.
   - To review the details and code of this update, please navigate to the `adaptive_sample_selection` folder.

2. **CNN Network Optimization**:
   - Replaced standard convolutions with depthwise separable convolutions, reduced initial channel sizes, removed dropout in convolutional layers, and added global average pooling for efficiency.
   - This change can be explored in the `changed_CNN_Model` folder.

## Results

- **Adaptive Sample Selection**: Increased accuracy on CIFAR-10 across test seeds, with a reduction in negative log-likelihood.
- **CNN Optimization**: Training time was reduced by over 40% per epoch.

Pre-trained model weights and further instructions are provided in each folder. 

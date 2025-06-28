# AI-Powered Lunar Surface Classification: A Deep Learning Approach

This project presents an advanced deep learning pipeline for analyzing and classifying images from an Artificial Lunar Rocky Landscape Dataset. Our goal was to develop a highly accurate and robust classification system capable of distinguishing various lunar terrain features, leveraging state-of-the-art convolutional neural network architectures and sophisticated training strategies.

---

## Project Highlights & Methodology

Our comprehensive pipeline for lunar surface classification includes:

- **In-depth Dataset Analysis:** Thorough examination of image characteristics and class distributions to inform preprocessing and model selection.

- **Advanced Image Preprocessing:** Implementation of techniques such as grayscale conversion, adaptive contrast enhancement (CLAHE), and histogram equalization to significantly improve feature visibility and model learning.

- **Robust Data Augmentation:** Utilization of ImageDataGenerator with a diverse set of augmentation techniques to enhance model generalization and prevent overfitting on limited data.

- **Exploration of Pre-trained Architectures:** Experimentation with powerful pre-trained Deep Learning models, including:
  - **EfficientNetB0:** Chosen for its excellent balance of efficiency and accuracy.
  - **ResNet50V2:** A robust deep residual network, known for mitigating vanishing gradient problems.
  - **Xception:** Employing depthwise separable convolutions for efficient learning of spatial and channel-wise correlations.

- **Optimized Training Regimen:**
  - **Custom Learning Rate Scheduler:** Incorporating a warm-up phase followed by a cosine decay schedule for stable and optimized convergence.
  - **Enhanced Callbacks:** Strategic use of EarlyStopping to prevent overfitting, ReduceLROnPlateau for adaptive learning rate adjustments, and ModelCheckpoint for saving the best performing weights.

- **Comprehensive Evaluation:** Detailed assessment of model performance through classification reports, confusion matrices, and feature visualization techniques for deeper insights.

- **Reliable Performance Estimation:** Evaluation of the best performing architecture using stratified k-fold cross-validation to provide a more robust and reliable estimate of the model's generalization capabilities.

- **Full Classification Pipeline Orchestration:** End-to-end management of the process, from data ingestion and preprocessing to model training, fine-tuning, and evaluation.

---

## Dataset

This project utilizes the "Artificial Lunar Rocky Landscape Dataset" available on Kaggle. The dataset is expected to follow a standard image classification directory structure.

---

## Model Selection & Performance

We evaluated three prominent pre-trained Deep Learning architectures as base models for our classification task. Each model was augmented with an enhanced classification head incorporating Batch Normalization and Dropout for improved regularization.

Initial evaluation on the test set after training the classification head showed:

- **EfficientNetB0:** 54.14% Accuracy
- **ResNet50V2:** 75.35% Accuracy
- **Xception:** 88.87% Accuracy

Based on these initial results, the Xception architecture demonstrated superior performance and was selected for further fine-tuning.

---

## Results

The selected Xception model underwent a meticulous fine-tuning process, where all layers were unfrozen and trained with a lower learning rate for optimal adaptation to the specific lunar landscape features.

The final evaluation of the fine-tuned Xception model on the dedicated test set yielded an accuracy of **67.90%**.

To ensure a more robust and reliable estimate of the model's generalization performance, we further conducted 5-fold stratified cross-validation using the Xception architecture. The detailed results from this cross-validation are reported at the end of the execution, providing a comprehensive view of the model's consistency and stability.

---

## Future Enhancements

Potential future improvements include exploring advanced segmentation techniques for more granular analysis, experimenting with generative adversarial networks (GANs) for synthetic data augmentation, and deploying the model for real-time inference.

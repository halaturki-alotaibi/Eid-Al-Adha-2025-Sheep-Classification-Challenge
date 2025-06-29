# 🐑 Eid Al-Adha 2025 - Sheep Classification Challenge

Welcome to my solution for the [Sheep Classification Challenge 2025 on Kaggle](https://www.kaggle.com/competitions/sheep-classification-challenge-2025), a competition aiming to automate the classification of sheep based on images taken during the Eid season.

---

## 📌 Problem Statement

In this challenge, we are tasked with building a model that accurately classifies sheep into one of several predefined categories using image data.

---

## 🔍 Approach

### 🛠 Model Overview

This solution uses **ConvNeXt-Small**, a modern convolutional neural network architecture, fine-tuned using **PyTorch** and the `timm` library. The training pipeline is designed to maximize generalization using cross-validation, strong data augmentation, and model ensembling.

---

### 🔁 Step-by-Step Process

 **Data Preparation**
   - Loaded image paths and labels from `train_labels.csv`.
   - Encoded class names to numeric IDs (`label2id`, `id2label`).


 **Modeling with ConvNeXt**
   - Loaded pretrained **ConvNeXt-Small** from `timm`.
   - Replaced final classification layer with a custom `nn.Linear` .
   - Used `torch.cuda.amp` for mixed precision training.

**Training Strategy**
   - Optimizer: `AdamW`
   - Scheduler: `CosineAnnealingLR`
   - Loss: `CrossEntropyLoss` (optionally replaced by `FocalLoss`)
   - Trained on each fold and saved best model weights based on validation accuracy.

**Inference and Submission**
   - Ran predictions on test data using trained models from each fold.
   - Aggregated predictions using soft voting (average of probabilities).
   - Prepared a submission file in the required Kaggle format.

---
## 🧪 Tools & Libraries

- PyTorch & Torchvision
- `timm` (ConvNeXt model)
- CUDA AMP (mixed precision)

## 🏆 Results

| Model          | Public LB   |  private LB |
|----------------|-------------|-----------|
| ConvNeXt-Small | ~0.84       | ~0.89     | 


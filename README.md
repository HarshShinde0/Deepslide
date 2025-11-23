---
title: DeepSlide - Landslide Detection and Mapping Using Deep Learning Across Multi-Source Satellite Data and Geographic Regions
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
tags:
- streamlit
- pytorch
- deep-learning
- landslide-detection
pinned: false
short_description: Landslide detection using various deep learning models
license: apache-2.0
---

# Landslide Detection Models Demo

This Space demonstrates various deep learning models for landslide detection, using models trained with PyTorch. The models are served directly from our [Kaggle Models Repository](https://www.kaggle.com/models/harshshinde8/sims/)
 or [harshinde/DeepSlide_Models](https://huggingface.co/harshinde/DeepSlide_Models).

## Available Models
- DeepLabV3+
- DenseNet121
- EfficientNetB0
- InceptionResNetV2
- InceptionV4
- MiT-B1
- MobileNetV2
- ResNet34
- ResNeXt50_32X4D
- SE-ResNet50
- SE-ResNeXt50_32X4D
- SegFormer
- VGG16

## How to Use
1. Select a model from the sidebar
2. Upload one or more `.h5` files containing satellite imagery
3. View the landslide detection results and predictions
4. Download the results if needed

## Model Information
All models are trained on satellite imagery data and are optimized for landslide detection. Each model has its own strengths and characteristics, which are described in the app interface when you select them.

## Technical Details
- Python 3.9
- PyTorch 1.9.0
- Streamlit 1.28.0
- Models are automatically downloaded from HuggingFace [harshinde/DeepSlide_Models](https://huggingface.co/harshinde/DeepSlide_Models).

## Author
- Harsh Shinde
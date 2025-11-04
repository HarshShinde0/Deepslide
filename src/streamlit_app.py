import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from pathlib import Path
from model_downloader import ModelDownloader

# Import models
from .deeplabv3plus_model import LandslideModel as DeepLabV3PlusModel
from .vgg16_model import LandslideModel as VGG16Model
from .resnet34_model import LandslideModel as ResNet34Model
from .efficientnetb0_model import LandslideModel as EfficientNetB0Model
from .mitb1_model import LandslideModel as MiTB1Model
from .inceptionv4_model import LandslideModel as InceptionV4Model
from .densenet121_model import LandslideModel as DenseNet121Model
from .resnext50_32x4d_model import LandslideModel as ResNeXt50_32X4DModel
from .se_resnet50_model import LandslideModel as SEResNet50Model
from .se_resnext50_32x4d_model import LandslideModel as SEResNeXt50_32X4DModel
from .segformer_model import LandslideModel as SegFormerB2Model
from .inceptionresnetv2_model import LandslideModel as InceptionResNetV2Model

# Initialize model downloader
model_downloader = ModelDownloader()

# Model descriptions
model_descriptions = {
    "MobileNetV2": {"type": "mobilenet_v2", "description": "MobileNetV2 is a lightweight deep learning model for image classification and segmentation."},
    "VGG16": {"type": "vgg16", "description": "VGG16 is a popular deep learning model known for its simplicity and depth."},
    "ResNet34": {"type": "resnet34", "description": "ResNet34 is a deep residual network that helps in training very deep networks."},
    "EfficientNetB0": {"type": "efficientnet_b0", "description": "EfficientNetB0 is part of the EfficientNet family, known for its efficiency and performance."},
    "MiT-B1": {"type": "mit_b1", "description": "MiT-B1 is a transformer-based model designed for segmentation tasks."},
    "InceptionV4": {"type": "inceptionv4", "description": "InceptionV4 is a convolutional neural network known for its inception modules."},
    "DeepLabV3+": {"type": "deeplabv3plus", "description": "DeepLabV3+ is an advanced model for semantic image segmentation."},
    "DenseNet121": {"type": "densenet121", "description": "DenseNet121 is a densely connected convolutional network for image classification and segmentation."},
    "ResNeXt50_32X4D": {"type": "resnext50_32x4d", "description": "ResNeXt50_32X4D is a highly modularized network aimed at improving accuracy."},
    "SEResNet50": {"type": "se_resnet50", "description": "SEResNet50 is a ResNet model with squeeze-and-excitation blocks for better feature recalibration."},
    "SEResNeXt50_32X4D": {"type": "se_resnext50_32x4d", "description": "SEResNeXt50_32X4D combines ResNeXt and SE blocks for improved performance."},
    "SegFormerB2": {"type": "segformer", "description": "SegFormerB2 is a transformer-based model for semantic segmentation."},
    "InceptionResNetV2": {"type": "inceptionresnetv2", "description": "InceptionResNetV2 is a hybrid model combining Inception and ResNet architectures."},
}

# Streamlit app
st.set_page_config(page_title="Landslide Detection", layout="wide")

st.title("Landslide Detection")
st.markdown("""
## Instructions
1. Select a model from the sidebar.
2. Upload one or more `.h5` files.
3. The app will process the files and display the input image, prediction, and overlay.
4. You can download the prediction results.
""")

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_type = st.sidebar.selectbox("Select Model", list(model_descriptions.keys()))

# Get model details
config = {
    'model_config': {
        'model_type': model_descriptions[model_type]['type'],
        'in_channels': 14,
        'num_classes': 1
    }
}

# Show model description
st.sidebar.markdown(f"**Model Type:** {model_descriptions[model_type]['type']}")
st.sidebar.markdown(f"**Description:** {model_descriptions[model_type]['description']}")

try:
    # Get the appropriate model class
    if model_type == "DeepLabV3+":
        model_class = DeepLabV3PlusModel
    else:
        model_class = locals()[model_type.replace("-", "") + "Model"]
    
    # Get model path from downloader
    model_name = model_descriptions[model_type]['type'].replace("+", "plus").lower()
    model_path = model_downloader.get_model_path(model_name)
    st.success(f"Model {model_type} loaded successfully!")

    # File uploader
    uploaded_files = st.file_uploader("Upload H5 files", type=['h5'], accept_multiple_files=True)
    
    if uploaded_files:
        # Process each uploaded file
        for uploaded_file in uploaded_files:
            st.write(f"Processing {uploaded_file.name}...")
            # Add your file processing logic here
            
except FileNotFoundError as e:
    st.error(f"Model file not found: {str(e)}")
    st.error("Please ensure all model files are present in the models directory")
    st.stop()
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.stop()
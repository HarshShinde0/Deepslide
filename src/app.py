import streamlit as st
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

# Import models
from mobilenetv2_model import LandslideModel as MobileNetV2Model
from vgg16_model import LandslideModel as VGG16Model
from resnet34_model import LandslideModel as ResNet34Model
from efficientnetb0_model import LandslideModel as EfficientNetB0Model
from mitb1_model import LandslideModel as MiTB1Model
from inceptionv4_model import LandslideModel as InceptionV4Model
from densenet121_model import LandslideModel as DenseNet121Model
from deeplabv3plus_model import LandslideModel as DeepLabV3PlusModel
from resnext50_32x4d_model import LandslideModel as ResNeXt50_32X4DModel
from se_resnet50_model import LandslideModel as SEResNet50Model
from se_resnext50_32x4d_model import LandslideModel as SEResNeXt50_32X4DModel
from segformer_model import LandslideModel as SegFormerB2Model
from inceptionresnetv2_model import LandslideModel as InceptionResNetV2Model

# Load the configuration file
config = """
model_config:
  model_type: "mobilenet_v2"
  in_channels: 14
  num_classes: 1
  encoder_weights: "imagenet"
  wce_weight: 0.5

dataset_config:
  num_classes: 1
  num_channels: 14
  channels: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
  normalize: False

train_config:
  dataset_path: ""
  checkpoint_path: "checkpoints"
  seed: 42
  train_val_split: 0.8
  batch_size: 16
  num_epochs: 100
  lr: 0.001
  device: "cuda:0"
  save_config: True
  experiment_name: "mobilenet_v2"

logging_config:
  wandb_project: "l4s"
  wandb_entity: "Silvamillion"
"""

config = yaml.safe_load(config)

# Model descriptions
model_descriptions = {
    "MobileNetV2": {"path": "mobilenetv2.pth", "type": "mobilenet_v2", "description": "MobileNetV2 is a lightweight deep learning model for image classification and segmentation."},
    "VGG16": {"path": "vgg16.pth", "type": "vgg16", "description": "VGG16 is a popular deep learning model known for its simplicity and depth."},
    "ResNet34": {"path": "resnet34.pth", "type": "resnet34", "description": "ResNet34 is a deep residual network that helps in training very deep networks."},
    "EfficientNetB0": {"path": "effucientnetb0.pth", "type": "efficientnet_b0", "description": "EfficientNetB0 is part of the EfficientNet family, known for its efficiency and performance."},
    "MiT-B1": {"path": "mitb1.pth", "type": "mit_b1", "description": "MiT-B1 is a transformer-based model designed for segmentation tasks."},
    "InceptionV4": {"path": "inceptionv4.pth", "type": "inceptionv4", "description": "InceptionV4 is a convolutional neural network known for its inception modules."},
    "DeepLabV3+": {"path": "deeplabv3.pth", "type": "deeplabv3+", "description": "DeepLabV3+ is an advanced model for semantic image segmentation."},
    "DenseNet121": {"path": "densenet121.pth", "type": "densenet121", "description": "DenseNet121 is a densely connected convolutional network for image classification and segmentation."},
    "ResNeXt50_32X4D": {"path": "resnext50-32x4d.pth", "type": "resnext50_32x4d", "description": "ResNeXt50_32X4D is a highly modularized network aimed at improving accuracy."},
    "SEResNet50": {"path": "se_resnet50.pth", "type": "se_resnet50", "description": "SEResNet50 is a ResNet model with squeeze-and-excitation blocks for better feature recalibration."},
    "SEResNeXt50_32X4D": {"path": "se_resnext50_32x4d.pth", "type": "se_resnext50_32x4d", "description": "SEResNeXt50_32X4D combines ResNeXt and SE blocks for improved performance."},
    "SegFormerB2": {"path": "segformer.pth", "type": "segformer_b2", "description": "SegFormerB2 is a transformer-based model for semantic segmentation."},
    "InceptionResNetV2": {"path": "inceptionresnetv2.pth", "type": "inceptionresnetv2", "description": "InceptionResNetV2 is a hybrid model combining Inception and ResNet architectures."},
}

# Streamlit app
st.set_page_config(page_title="Landslide Detection", layout="wide")

st.title("Landslide Detection")
st.markdown("""
## Instructions
1. Select a model from the sidebar or choose to run all models.
2. Upload one or more `.h5` files.
3. The app will process the files and display the input image, prediction, and overlay.
4. You can download the prediction results.
""")

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_option = st.sidebar.radio("Choose an option", ["Select a single model", "Run all models"])
if model_option == "Select a single model":
    model_type = st.sidebar.selectbox("Select Model", list(model_descriptions.keys()))
    config['model_config']['model_type'] = model_descriptions[model_type]['type']
    if model_type == "DeepLabV3+":
        model_class = DeepLabV3PlusModel
    else:
        model_class = locals()[model_type.replace("-", "") + "Model"]
    model_path = model_descriptions[model_type]['path']

    # Display model details in the sidebar
    st.sidebar.markdown(f"**Model Type:** {model_descriptions[model_type]['type']}")
    st.sidebar.markdown(f"**Model Path:** {model_descriptions[model_type]['path']}")
    st.sidebar.markdown(f"**Description:** {model_descriptions[model_type]['description']}")

# Main content
st.header("Upload Data")
uploaded_files = st.file_uploader("Choose .h5 files...", type="h5", accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"Processing file: {uploaded_file.name}")
        with st.spinner('Classifying...'):
            with h5py.File(uploaded_file, 'r') as hdf:
                data = np.array(hdf.get('img'))
                data[np.isnan(data)] = 0.000001
                channels = config["dataset_config"]["channels"]
                image = np.zeros((128, 128, len(channels)))
                for i, channel in enumerate(channels):
                    image[:, :, i] = data[:, :, channel-1]

            # Transform the image to the required format
            image = image.transpose((2, 0, 1))  # (H, W, C) to (C, H, W)
            image = torch.from_numpy(image).float().unsqueeze(0)  # Add batch dimension

            if model_option == "Select a single model":
                # Process the image with the selected model
                st.write(f"Using model: {model_type}")

                # Load the model
                model = model_class(config)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
                model.eval()

                # Make prediction
                with torch.no_grad():
                    prediction = model(image)
                    prediction = torch.sigmoid(prediction).cpu().numpy()

                # Display prediction
                st.header(f"Prediction Results - {model_type}")
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                img = image.squeeze().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())  # Normalize the image to [0, 1] range for display
                ax[0].imshow(img[:, :, 1:4])  # Display first three channels as RGB
                ax[0].set_title("Input Image")
                ax[1].imshow(prediction.squeeze() > 0.5, cmap='plasma')  # Apply threshold
                ax[1].set_title("Prediction")
                ax[2].imshow(img[:, :, :3])  # Display first three channels as RGB
                ax[2].imshow(prediction.squeeze() > 0.5, cmap='plasma', alpha=0.3)  # Overlay prediction
                ax[2].set_title("Overlay")
                st.pyplot(fig)

                # Option to download the prediction
                st.write(f"Download the prediction as a .npy file for {model_type}:")
                npy_data = prediction.squeeze()
                st.download_button(
                    label=f"Download Prediction - {model_type}",
                    data=npy_data.tobytes(),
                    file_name=f"{uploaded_file.name.split('.')[0]}_{model_type}_prediction.npy",
                    mime="application/octet-stream"
                )

            else:
                # Process the image with each model
                for model_name, model_info in model_descriptions.items():
                    st.write(f"Using model: {model_name}")
                    if model_name == "DeepLabV3+":
                        model_class = DeepLabV3PlusModel
                    else:
                        model_class = locals()[model_name.replace("-", "") + "Model"]
                    model_path = model_info['path']
                    config['model_config']['model_type'] = model_info['type']

                    # Load the model
                    model = model_class(config)
                    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
                    model.eval()

                    # Make prediction
                    with torch.no_grad():
                        prediction = model(image)
                        prediction = torch.sigmoid(prediction).cpu().numpy()

                    # Display prediction
                    st.header(f"Prediction Results - {model_name}")
                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                    img = image.squeeze().permute(1, 2, 0).numpy()
                    img = (img - img.min()) / (img.max() - img.min())  # Normalize the image to [0, 1] range for display
                    ax[0].imshow(img[:, :, :3])  # Display first three channels as RGB
                    ax[0].set_title("Input Image")
                    ax[1].imshow(prediction.squeeze() > 0.5, cmap='plasma')  # Apply threshold
                    ax[1].set_title("Prediction")
                    ax[2].imshow(img[:, :, :3])  # Display first three channels as RGB
                    ax[2].imshow(prediction.squeeze() > 0.5, cmap='plasma', alpha=0.3)  # Overlay prediction
                    ax[2].set_title("Overlay")
                    st.pyplot(fig)

                    # Option to download the prediction
                    st.write(f"Download the prediction as a .npy file for {model_name}:")
                    npy_data = prediction.squeeze()
                    st.download_button(
                        label=f"Download Prediction - {model_name}",
                        data=npy_data.tobytes(),
                        file_name=f"{uploaded_file.name.split('.')[0]}_{model_name}_prediction.npy",
                        mime="application/octet-stream"
                    )

st.success('Done!')

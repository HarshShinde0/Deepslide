import streamlit as st
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

# Import models
from src.mobilenetv2_model import LandslideModel as MobileNetV2Model
from src.vgg16_model import LandslideModel as VGG16Model
from src.resnet34_model import LandslideModel as ResNet34Model
from src.efficientnetb0_model import LandslideModel as EfficientNetB0Model
from src.mitb1_model import LandslideModel as MiTB1Model
from src.inceptionv4_model import LandslideModel as InceptionV4Model
from src.densenet121_model import LandslideModel as DenseNet121Model
from src.deeplabv3plus_model import LandslideModel as DeepLabV3PlusModel
from src.resnext50_32x4d_model import LandslideModel as ResNeXt50_32X4DModel
from src.se_resnet50_model import LandslideModel as SEResNet50Model
from src.se_resnext50_32x4d_model import LandslideModel as SEResNeXt50_32X4DModel
from src.segformer_model import LandslideModel as SegFormerB2Model
from src.inceptionresnetv2_model import LandslideModel as InceptionResNetV2Model

# Define available models
AVAILABLE_MODELS = {
    "mobilenetv2": {"name": "MobileNetV2", "type": "mobilenet_v2"},
    "vgg16": {"name": "VGG16", "type": "vgg16"},
    "resnet34": {"name": "ResNet34", "type": "resnet34"},
    "efficientnetb0": {"name": "EfficientNetB0", "type": "efficientnet_b0"},
    "mitb1": {"name": "MiTB1", "type": "mitb1"},
    "inceptionv4": {"name": "InceptionV4", "type": "inception_v4"},
    "densenet121": {"name": "DenseNet121", "type": "densenet121"},
    "deeplabv3plus": {"name": "DeepLabV3Plus", "type": "deeplabv3plus"},
    "resnext50": {"name": "ResNeXt50", "type": "resnext50_32x4d"},
    "seresnet50": {"name": "SEResNet50", "type": "se_resnet50"},
    "seresnext50": {"name": "SEResNeXt50", "type": "se_resnext50_32x4d"},
    "segformerb2": {"name": "SegFormerB2", "type": "segformer_b2"},
    "inceptionresnetv2": {"name": "InceptionResNetV2", "type": "inception_resnet_v2"}
}

# Model descriptions with their respective types and descriptions
MODEL_DESCRIPTIONS = {
    model_key: {
        "type": model_info["type"],
        "description": f"{model_info['name']} - A model for landslide detection and segmentation.",
        "name": model_info["name"]
    }
    for model_key, model_info in AVAILABLE_MODELS.items()
}

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
"""

config = yaml.safe_load(config)

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
    selected_model_key = st.sidebar.selectbox("Select Model", list(MODEL_DESCRIPTIONS.keys()))
    selected_model_info = MODEL_DESCRIPTIONS[selected_model_key]
    config['model_config']['model_type'] = selected_model_info['type']
    
    # Display model details in the sidebar
    st.sidebar.markdown("### Model Details")
    st.sidebar.markdown(f"**Model Name:** {selected_model_info['name']}")
    st.sidebar.markdown(f"**Model Type:** {selected_model_info['type']}")
    st.sidebar.markdown(f"**Description:** {selected_model_info['description']}")

# Main content
st.header("Upload Data")

# Initialize session state for error tracking if not exists
if 'upload_errors' not in st.session_state:
    st.session_state.upload_errors = []

uploaded_files = st.file_uploader(
    "Choose .h5 files...", 
    type="h5", 
    accept_multiple_files=True,
    help="Upload your .h5 files here. Maximum file size is 200MB."
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"Processing file: {uploaded_file.name}")
        st.write(f"File size: {uploaded_file.size} bytes")
        
        with st.spinner('Classifying...'):
            try:
                # Read the file directly using BytesIO
                import io
                bytes_data = uploaded_file.getvalue()
                bytes_io = io.BytesIO(bytes_data)
                
                with h5py.File(bytes_io, 'r') as hdf:
                    if 'img' not in hdf:
                        st.error(f"Error: 'img' dataset not found in {uploaded_file.name}")
                        continue
                        
                    data = np.array(hdf.get('img'))
                    data[np.isnan(data)] = 0.000001
                    channels = config["dataset_config"]["channels"]
                    image = np.zeros((128, 128, len(channels)))

                    for i, band in enumerate(channels):
                        image[:, :, i] = data[band-1]

                    selected_channels = [image[:, :, i] for i in range(3)]
                    image = np.transpose(image, (2, 0, 1))

                    if model_option == "Select a single model":
                        # Get the model class from AVAILABLE_MODELS
                        model_class_name = AVAILABLE_MODELS[selected_model_key]['name'].replace('-', '') + 'Model'
                        model_class = locals()[model_class_name]

                        # Initialize model downloader
                        from model_downloader import ModelDownloader
                        downloader = ModelDownloader()
                        
                        try:
                            # Download/get model path
                            model_path = downloader.download_model(selected_model_key)
                            st.info(f"Using model from: {model_path}")
                            
                            # Load the model
                            model = model_class(config)
                            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
                            model.eval()

                            # Make prediction
                            with torch.no_grad():
                                prediction = model(torch.from_numpy(image).unsqueeze(0).float())
                                prediction = torch.sigmoid(prediction).numpy()

                            st.header(f"Prediction Results - {selected_model_info['name']}")

                            # Create columns for input image, prediction, and overlay
                            col1, col2, col3 = st.columns(3)

                            # Display input image
                            with col1:
                                st.write("Input Image")
                                plt.figure(figsize=(8, 8))
                                plt.imshow(selected_channels[0], cmap='viridis')
                                plt.colorbar()
                                plt.axis('off')
                                st.pyplot(plt)

                            # Display prediction
                            with col2:
                                st.write("Prediction")
                                plt.figure(figsize=(8, 8))
                                plt.imshow(prediction.squeeze(), cmap='viridis')
                                plt.colorbar()
                                plt.axis('off')
                                st.pyplot(plt)

                            # Display overlay
                            with col3:
                                st.write("Overlay")
                                plt.figure(figsize=(8, 8))
                                plt.imshow(selected_channels[0], cmap='viridis')
                                plt.imshow(prediction.squeeze(), cmap='viridis', alpha=0.5)
                                plt.colorbar()
                                plt.axis('off')
                                st.pyplot(plt)

                            # Download button for prediction
                            st.write(f"Download the prediction as a .npy file for {selected_model_info['name']}:")
                            npy_data = prediction.squeeze()
                            st.download_button(
                                label=f"Download Prediction - {selected_model_info['name']}",
                                data=npy_data.tobytes(),
                                file_name=f"{uploaded_file.name.split('.')[0]}_{selected_model_key}_prediction.npy",
                                mime="application/octet-stream"
                            )

                        except Exception as e:
                            st.error(f"Error with model {selected_model_info['name']}: {str(e)}")
                    else:
                        # Process the image with each model
                        for model_key, model_info in MODEL_DESCRIPTIONS.items():
                            st.write(f"Using model: {model_info['name']}")
                            config['model_config']['model_type'] = model_info['type']
                            
                            # Get the model class from AVAILABLE_MODELS
                            model_class_name = AVAILABLE_MODELS[model_key]['name'].replace('-', '') + 'Model'
                            model_class = locals()[model_class_name]

                            # Initialize model downloader
                            from model_downloader import ModelDownloader
                            downloader = ModelDownloader()
                            
                            try:
                                # Download/get model path
                                model_path = downloader.download_model(model_key)
                                st.info(f"Using model from: {model_path}")
                                
                                # Load the model
                                model = model_class(config)
                                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
                                model.eval()

                                # Make prediction
                                with torch.no_grad():
                                    prediction = model(torch.from_numpy(image).unsqueeze(0).float())
                                    prediction = torch.sigmoid(prediction).numpy()

                                st.header(f"Prediction Results - {model_info['name']}")

                                # Create columns for input image, prediction, and overlay
                                col1, col2, col3 = st.columns(3)

                                # Display input image
                                with col1:
                                    st.write("Input Image")
                                    plt.figure(figsize=(8, 8))
                                    plt.imshow(selected_channels[0], cmap='viridis')
                                    plt.colorbar()
                                    plt.axis('off')
                                    st.pyplot(plt)

                                # Display prediction
                                with col2:
                                    st.write("Prediction")
                                    plt.figure(figsize=(8, 8))
                                    plt.imshow(prediction.squeeze(), cmap='viridis')
                                    plt.colorbar()
                                    plt.axis('off')
                                    st.pyplot(plt)

                                # Display overlay
                                with col3:
                                    st.write("Overlay")
                                    plt.figure(figsize=(8, 8))
                                    plt.imshow(selected_channels[0], cmap='viridis')
                                    plt.imshow(prediction.squeeze(), cmap='viridis', alpha=0.5)
                                    plt.colorbar()
                                    plt.axis('off')
                                    st.pyplot(plt)

                                # Download button for prediction
                                st.write(f"Download the prediction as a .npy file for {model_info['name']}:")
                                npy_data = prediction.squeeze()
                                st.download_button(
                                    label=f"Download Prediction - {model_info['name']}",
                                    data=npy_data.tobytes(),
                                    file_name=f"{uploaded_file.name.split('.')[0]}_{model_key}_prediction.npy",
                                    mime="application/octet-stream"
                                )

                            except Exception as e:
                                st.error(f"Error with model {model_info['name']}: {str(e)}")
                                continue

            except Exception as e:
                st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
                continue
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

# Import models
from src.mobilenetv2_model import LandslideModel as MobileNetV2Model
from src.vgg16_model import LandslideModel as VGG16Model
from src.resnet34_model import LandslideModel as ResNet34Model
from src.efficientnetb0_model import LandslideModel as EfficientNetB0Model
from src.mitb1_model import LandslideModel as MiTB1Model
from src.inceptionv4_model import LandslideModel as InceptionV4Model
from src.densenet121_model import LandslideModel as DenseNet121Model
from src.deeplabv3plus_model import LandslideModel as DeepLabV3PlusModel
from src.resnext50_32x4d_model import LandslideModel as ResNeXt50_32X4DModel
from src.se_resnet50_model import LandslideModel as SEResNet50Model
from src.se_resnext50_32x4d_model import LandslideModel as SEResNeXt50_32X4DModel
from segformer_model import LandslideModel as SegFormerB2Model
from inceptionresnetv2_model import LandslideModel as InceptionResNetV2Model

# Define available models
AVAILABLE_MODELS = {
    "mobilenetv2": {"name": "MobileNetV2", "type": "mobilenet_v2"},
    "vgg16": {"name": "VGG16", "type": "vgg16"},
    "resnet34": {"name": "ResNet34", "type": "resnet34"},
    "efficientnetb0": {"name": "EfficientNetB0", "type": "efficientnet_b0"},
    "mitb1": {"name": "MiTB1", "type": "mitb1"},
    "inceptionv4": {"name": "InceptionV4", "type": "inception_v4"},
    "densenet121": {"name": "DenseNet121", "type": "densenet121"},
    "deeplabv3plus": {"name": "DeepLabV3Plus", "type": "deeplabv3plus"},
    "resnext50": {"name": "ResNeXt50", "type": "resnext50_32x4d"},
    "seresnet50": {"name": "SEResNet50", "type": "se_resnet50"},
    "seresnext50": {"name": "SEResNeXt50", "type": "se_resnext50_32x4d"},
    "segformerb2": {"name": "SegFormerB2", "type": "segformer_b2"},
    "inceptionresnetv2": {"name": "InceptionResNetV2", "type": "inception_resnet_v2"}
}

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

# Model descriptions with their respective types and descriptions
MODEL_DESCRIPTIONS = {
    model_key: {
        "type": model_info["type"],
        "description": f"{model_info['name']} - A model for landslide detection and segmentation.",
        "name": model_info["name"]
    }
    for model_key, model_info in AVAILABLE_MODELS.items()
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
    selected_model = st.sidebar.selectbox("Select Model", list(MODEL_DESCRIPTIONS.keys()))
    config['model_config']['model_type'] = MODEL_DESCRIPTIONS[selected_model]['type']
    
    # Display model details in the sidebar
    st.sidebar.markdown(f"**Model Name:** {MODEL_DESCRIPTIONS[selected_model]['name']}")
    st.sidebar.markdown(f"**Model Type:** {MODEL_DESCRIPTIONS[selected_model]['type']}")
    st.sidebar.markdown(f"**Description:** {MODEL_DESCRIPTIONS[selected_model]['description']}")

# Main content
st.header("Upload Data")

# Initialize session state for error tracking if not exists
if 'upload_errors' not in st.session_state:
    st.session_state.upload_errors = []

uploaded_files = st.file_uploader(
    "Choose .h5 files...", 
    type="h5", 
    accept_multiple_files=True,
    help="Upload your .h5 files here. Maximum file size is 200MB."
)

if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"Processing file: {uploaded_file.name}")
            
            # Display file details for debugging
            st.write(f"File size: {uploaded_file.size} bytes")
            
            with st.spinner('Classifying...'):
                try:
                    # Read the file directly using BytesIO
                    import io
                    bytes_data = uploaded_file.getvalue()
                    bytes_io = io.BytesIO(bytes_data)
                    
                    with h5py.File(bytes_io, 'r') as hdf:
                        # Check if 'img' exists in the file
                        if 'img' not in hdf:
                            st.error(f"Error: 'img' dataset not found in {uploaded_file.name}")
                            continue
                            
                        data = np.array(hdf.get('img'))
                        data[np.isnan(data)] = 0.000001
                        channels = config["dataset_config"]["channels"]
                        image = np.zeros((128, 128, len(channels)))
                        
                except h5py.Error as he:
                    st.error(f"H5PY Error processing {uploaded_file.name}: {str(he)}")
                    continue
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    continue
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
                for model_key, model_info in MODEL_DESCRIPTIONS.items():
                    st.write(f"Using model: {model_info['name']}")
                    config['model_config']['model_type'] = model_info['type']
                    
                    # Get the model class from AVAILABLE_MODELS
                    model_class_name = AVAILABLE_MODELS[model_key]['name'].replace('-', '') + 'Model'
                    model_class = locals()[model_class_name]

                    # Initialize model downloader
                    from model_downloader import ModelDownloader
                    downloader = ModelDownloader()
                    
                    try:
                        # Download/get model path
                        model_path = downloader.download_model(model_name.lower())
                        st.info(f"Using model from: {model_path}")
                        
                        # Load the model
                        model = model_class(config)
                        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
                        model.eval()
                    except Exception as e:
                        st.error(f"Error loading model {model_name}: {str(e)}")
                        continue

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

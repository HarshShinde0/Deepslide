import os
import json
import requests
import streamlit as st
from pathlib import Path
from tqdm.auto import tqdm

class ModelDownloader:
    def __init__(self):
        # Create models directory for caching
        self.models_dir = Path("/app/models")
        self.models_dir.mkdir(exist_ok=True)
        
        # HuggingFace model repository details
        self.hf_model_url = "https://huggingface.co/harshinde/Sims/resolve/main/models/"
        
        # Model mapping with file names
        self.model_files = {
            "deeplabv3plus": {
                "file": "deeplabv3.pth",
                "url": f"{self.hf_model_url}deeplabv3.pth"
            },
            "densenet121": {
                "file": "densenet121.pth",
                "url": f"{self.hf_model_url}densenet121.pth"
            },
            "efficientnetb0": {
                "file": "efficientnetb0.pth",
                "url": f"{self.hf_model_url}efficientnetb0.pth"
            },
            "inceptionresnetv2": {
                "file": "inceptionresnetv2.pth",
                "id": "inceptionresnetv2"
            },
            "inceptionv4": {
                "file": "inceptionv4.pth",
                "id": "inceptionv4"
            },
            "mitb1": {
                "file": "mitb1.pth",
                "id": "mitb1"
            },
            "mobilenetv2": {
                "file": "mobilenetv2.pth",
                "id": "mobilenetv2"
            },
            "resnet34": {
                "file": "resnet34.pth",
                "id": "resnet34"
            },
            "resnext50_32x4d": {
                "file": "resnext50-32x4d.pth",
                "id": "resnext50-32x4d"
            },
            "se_resnet50": {
                "file": "se_resnet50.pth",
                "id": "se_resnet50"
            },
            "se_resnext50_32x4d": {
                "file": "se_resnext50_32x4d.pth",
                "id": "se_resnext50_32x4d"
            },
            "segformer": {
                "file": "segformer.pth",
                "id": "segformer"
            },
            "vgg16": {
                "file": "vgg16.pth",
                "id": "vgg16"
            }
        }

    def download_model(self, model_name):
        """
        Download model from Hugging Face Models repository
        Args:
            model_name (str): Name of the model to download
        Returns:
            str: Path to the downloaded model file
        """
        if model_name not in self.model_files:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.model_files.keys())}")

        model_info = self.model_files[model_name]
        model_path = self.models_dir / model_info['file']

        if not model_path.exists():
            print(f"Downloading {model_name} model...")
            response = requests.get(model_info['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(model_path, 'wb') as f, tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            print(f"Model downloaded successfully to {model_path}")
        
        return str(model_path)
        
        # If model already exists, return path
        if model_path.exists():
            return str(model_path)

        # Construct download URL for the specific model
        download_url = f"{self.kaggle_model_url}/{model_info['id']}/1"
        
        try:
            st.info(f"Downloading model {model_name} from Kaggle Models...")
            progress_bar = st.progress(0)
            
            # Download with progress
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            with open(model_path, 'wb') as f:
                for i, data in enumerate(response.iter_content(block_size)):
                    progress_bar.progress(min(i * block_size / total_size, 1.0))
                    f.write(data)
            
            st.success(f"Successfully downloaded {model_name}")
            return str(model_path)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download model from Kaggle: {str(e)}")

    def get_model_path(self, model_name):
        """
        Get the path for a model file, downloading it from Kaggle if necessary
        Args:
            model_name (str): Name of the model (e.g., 'deeplabv3plus', 'densenet121', etc.)
        Returns:
            str: Path to the model file
        """
        if model_name not in self.model_files:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.model_files.keys())}")
        
        model_info = self.model_files[model_name]
        model_path = self.models_dir / model_info['file']
        
        # If model doesn't exist locally, download it
        if not model_path.exists():
            return self.download_from_kaggle(model_name)
        
        return str(model_path)

    def list_available_models(self):
        """
        List all available models
        Returns:
            list: List of available model names
        """
        return list(self.model_files.keys())
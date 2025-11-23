import os
import sys

# Add the parent directory to sys.path to allow imports from 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_downloader import ModelDownloader


def download_all():
    print("Starting download of all models...")
    downloader = ModelDownloader()
    models = downloader.list_available_models()

    for model_name in models:
        try:
            print(f"Checking/Downloading {model_name}...")
            path = downloader.download_model(model_name)
            print(f"✓ {model_name} is ready at {path}")
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")

    print("\nAll downloads completed.")


if __name__ == "__main__":
    download_all()

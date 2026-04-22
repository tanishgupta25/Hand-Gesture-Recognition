import os
import subprocess

def download_dataset():
    """
    Downloads the Sign Language Digits Dataset from GitHub.
    This dataset contains images of hands showing digits 0-9.
    """
    repo_url = "https://github.com/ardamavi/Sign-Language-Digits-Dataset.git"
    target_dir = "dataset_repo"

    print("Checking if dataset exists...")
    if not os.path.exists(target_dir):
        print(f"Downloading dataset from {repo_url}...")
        try:
            subprocess.run(["git", "clone", repo_url, target_dir], check=True)
            print("Dataset downloaded successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading dataset: {e}")
            return
    else:
        print("Dataset already downloaded.")
    
    # The actual images are inside dataset_repo/Dataset
    dataset_path = os.path.join(target_dir, "Dataset")
    if os.path.exists(dataset_path):
        print(f"Dataset is ready at: {os.path.abspath(dataset_path)}")
        print("Folder structure:")
        for digit in range(10):
            folder = os.path.join(dataset_path, str(digit))
            if os.path.exists(folder):
                num_images = len(os.listdir(folder))
                print(f" - Digit {digit}: {num_images} images")
    else:
        print("Error: Could not find 'Dataset' folder inside the repository.")

if __name__ == "__main__":
    download_dataset()

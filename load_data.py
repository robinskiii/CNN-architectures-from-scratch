import os

import kagglehub

#####################################

def download_data() -> str :
    print("Attempting to locate dataset via kagglehub...")
    try:
        path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    except (ConnectionError, OSError, ValueError) as e:
        print(f"{e}Offline mode: Could not reach Kaggle API. Checking local cache...")

        base_cache = os.environ.get("KAGGLEHUB_CACHE", os.path.expanduser("~/.cache/kagglehub"))
        dataset_cache = os.path.join(base_cache, "datasets", "jessicali9530", "celeba-dataset", "versions")

        if os.path.exists(dataset_cache):

            versions = [d for d in os.listdir(dataset_cache) if os.path.isdir(os.path.join(dataset_cache, d))]
            if versions:
                latest_version = max(versions, key=lambda x: int(x) if x.isdigit() else 0)[-1]
                path = os.path.join(dataset_cache, latest_version)
                print(f"Successfully found cached dataset at: {path}")
            else:
                raise FileNotFoundError("Local cache found, but it is empty. Connect to Wi-Fi to download.")
        else:
            raise FileNotFoundError("Dataset not found locally. You must connect to Wi-Fi to download it at least once.")

    print("Path to dataset files:", path)
    return path


if __name__== '_main_':
    download_data()

import os
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from PIL import Image

#####################################

DEFAULT_ATTRIBUTE = "Smiling"
DATASET_NAME = "jessicali9530/celeba-dataset"


def download_data() -> str:
    print("Attempting to locate dataset via kagglehub...")
    try:
        path = kagglehub.dataset_download(DATASET_NAME)
    except (ConnectionError, OSError, ValueError) as e:
        print(f"{e}Offline mode: Could not reach Kaggle API. Checking local cache...")

        base_cache = os.environ.get("KAGGLEHUB_CACHE", os.path.expanduser("~/.cache/kagglehub"))
        dataset_cache = os.path.join(base_cache, "datasets", "jessicali9530", "celeba-dataset", "versions")

        if os.path.exists(dataset_cache):
            versions = [
                d for d in os.listdir(dataset_cache)
                if os.path.isdir(os.path.join(dataset_cache, d))
            ]
            if versions:
                latest_version = max(versions, key=lambda x: int(x) if x.isdigit() else -1)
                path = os.path.join(dataset_cache, latest_version)
                print(f"Successfully found cached dataset at: {path}")
            else:
                raise FileNotFoundError("Local cache found, but it is empty. Connect to Wi-Fi to download.")
        else:
            raise FileNotFoundError("Dataset not found locally. You must connect to Wi-Fi to download it at least once.")

    print("Path to dataset files:", path)
    return path


def process_data(
    attribute_name: str = DEFAULT_ATTRIBUTE,
    max_samples: int | None = None,
    image_size: tuple[int, int] | None = None,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CelebA images, binary labels, and image IDs into NumPy arrays.

    The labels are derived from one attribute column in list_attr_celeba.csv
    Values in the CelebA attribute file are typically -1 and 1, so we map them to 0 and 1 respectively
    """
    data_root = Path(download_data())
    attr_csv_path = data_root / "list_attr_celeba.csv"
    partition_csv_path = data_root / "list_eval_partition.csv"
    images_folder_path = data_root / "img_align_celeba" / "img_align_celeba"

    if not attr_csv_path.exists():
        raise FileNotFoundError(f"Could not find attribute CSV at: {attr_csv_path}")
    if not images_folder_path.exists():
        raise FileNotFoundError(f"Could not find image folder at: {images_folder_path}")

    df_attr = pd.read_csv(attr_csv_path)

    if attribute_name not in df_attr.columns:
        available = ", ".join(df_attr.columns[1:11])
        raise ValueError(
            f"Attribute '{attribute_name}' was not found in CelebA metadata. "
            f"Try one of: {available}"
        )

    if partition_csv_path.exists():
        df_partition = pd.read_csv(partition_csv_path)
        df = df_attr.merge(df_partition, on="image_id", how="inner")
        df = df[df["partition"] == 0].copy()
    else:
        df = df_attr.copy()

    df = df[["image_id", attribute_name]].copy()

    if shuffle:
        df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)

    if max_samples is not None:
        df = df.head(max_samples).reset_index(drop=True)

    images: list[np.ndarray] = []
    labels: list[int] = []
    image_ids: list[str] = []

    for _, row in df.iterrows():
        image_filename = row["image_id"]
        label_value = row[attribute_name]

        full_image_path = images_folder_path / image_filename
        with Image.open(full_image_path) as img:
            img = img.convert("RGB")
            if image_size is not None:
                img = img.resize(image_size)

            image_array = np.asarray(img, dtype=np.float32) / 255.0 # normalising
            image_array = np.transpose(image_array, (2, 0, 1))

        images.append(image_array)
        labels.append(1 if int(label_value) == 1 else 0)
        image_ids.append(image_filename)

    if not images:
        raise ValueError("No training samples were loaded from the dataset.")

    X_train = np.stack(images, axis=0)
    y_train = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    image_ids_array = np.asarray(image_ids, dtype=str)

    return X_train, y_train, image_ids_array


if __name__ == "__main__":
    X_train, y_train, image_ids = process_data(max_samples=1000)
    print(f"Loaded X_train with shape {X_train.shape}")
    print(f"Loaded y_train with shape {y_train.shape}")
    print(f"Loaded image_ids with shape {image_ids.shape}")

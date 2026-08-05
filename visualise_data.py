import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from load_data import download_data

#####################################

def open_random_image(path) -> None:
    attr_csv_path = os.path.join(path, "list_attr_celeba.csv")
    images_folder_path = os.path.join(path, "img_align_celeba", "img_align_celeba")

    print("Loading attributes CSV...")
    df_attr = pd.read_csv(attr_csv_path)

    # PICKING RANDOM IMAGE FROM DATASET
    random_row = df_attr.sample(n=1).iloc[0]
    image_filename = random_row['image_id']

    # FINDING POSITIVE FEATURES
    stats = random_row.drop('image_id')
    positive_attrs = stats[stats == 1].index.tolist()

    # FEATURES OF CHOSEN IMAGE
    print("\n", 40*"=")
    print(9*"-", f" Stats for {image_filename} ", 9*"-")
    print(40*"=")
    print(f"Positive Attributes: {len(positive_attrs)}")
    print(", ".join(attrs.replace("_", " ") for attrs in positive_attrs) if positive_attrs else "None")
    print(40*"=")

    # EXTRACTING IMAGE FROM FOLDER
    try:
        full_image_path = os.path.join(images_folder_path, image_filename)
        img = Image.open(full_image_path)

        # RENDERING IMAGE
        plt.figure(figsize=(5, 6))
        plt.imshow(img)
        plt.title(f"{image_filename}\n({len(positive_attrs)} attributes)", fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find the image at {full_image_path}.")

if __name__== '__main__':

    # DOWNLOADING / LOCATING DATASET
    path = download_data()

    open_random_image(path)

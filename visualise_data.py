import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from load_data import DEFAULT_ATTRIBUTE, download_data

#####################################


@dataclass
class Sample:
    image_id: str
    label: int


def load_attribute_pools(path: str, attribute_name: str = DEFAULT_ATTRIBUTE) -> tuple[list[Sample], list[Sample]]:
    attr_csv_path = os.path.join(path, "list_attr_celeba.csv")
    partition_csv_path = os.path.join(path, "list_eval_partition.csv")

    print("Loading attributes CSV...")
    df_attr = pd.read_csv(attr_csv_path)

    if attribute_name not in df_attr.columns:
        available = ", ".join(df_attr.columns[1:11])
        raise ValueError(
            f"Attribute '{attribute_name}' was not found in CelebA metadata. "
            f"Try one of: {available}"
        )

    if os.path.exists(partition_csv_path):
        df_partition = pd.read_csv(partition_csv_path)
        df = df_attr.merge(df_partition, on="image_id", how="inner")
        df = df[df["partition"] == 0].copy()
    else:
        df = df_attr.copy()

    positive_rows = df[df[attribute_name] == 1]
    negative_rows = df[df[attribute_name] == -1]

    positive_samples = [Sample(row["image_id"], 1) for _, row in positive_rows.iterrows()]
    negative_samples = [Sample(row["image_id"], 0) for _, row in negative_rows.iterrows()]

    if not positive_samples:
        raise ValueError(f"No positive samples were found for attribute '{attribute_name}'.")
    if not negative_samples:
        raise ValueError(f"No negative samples were found for attribute '{attribute_name}'.")

    return positive_samples, negative_samples


def open_visualisation_stream(path: str, attribute_name: str = DEFAULT_ATTRIBUTE) -> None:
    images_folder_path = Path(path) / "img_align_celeba" / "img_align_celeba"
    positive_samples, negative_samples = load_attribute_pools(path, attribute_name=attribute_name)

    fig, ax = plt.subplots(figsize=(6, 7))
    import numpy as np

    rng = np.random.default_rng()
    positive_order = rng.permutation(len(positive_samples))
    negative_order = rng.permutation(len(negative_samples))
    positive_index = 0
    negative_index = 0

    def next_sample(label: int) -> Sample:
        nonlocal positive_order, negative_order, positive_index, negative_index

        if label == 1:
            if positive_index >= len(positive_order):
                positive_order = rng.permutation(len(positive_samples))
                positive_index = 0
            sample = positive_samples[int(positive_order[positive_index])]
            positive_index += 1
            return sample

        if negative_index >= len(negative_order):
            negative_order = rng.permutation(len(negative_samples))
            negative_index = 0
        sample = negative_samples[int(negative_order[negative_index])]
        negative_index += 1
        return sample

    def render(sample: Sample) -> None:
        full_image_path = images_folder_path / sample.image_id
        with Image.open(full_image_path) as img:
            img = img.convert("RGB")
            image = img.copy()

        ax.clear()
        ax.imshow(image)
        label_text = "Positive" if sample.label == 1 else "Negative"
        ax.set_title(
            f"{attribute_name}: {label_text}\n{sample.image_id}",
            fontsize=11,
        )
        ax.axis("off")
        fig.canvas.draw_idle()

        print(f"Showing {label_text.lower()} sample: {sample.image_id}")

    def on_key(event) -> None:
        if event.key == "up":
            render(next_sample(1))
        elif event.key == "down":
            render(next_sample(0))
        elif event.key in {"q", "escape"}:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("\n")
    print(104 * "=")
    print(f"Interactive stream for attribute: {attribute_name}")
    print("Press Up to show a positive sample.")
    print("Press Down to show a negative sample.")
    print("Press Q or Escape to quit.")
    print(104 * "=")

    render(next_sample(1))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = download_data()
    open_visualisation_stream(path, attribute_name=DEFAULT_ATTRIBUTE)

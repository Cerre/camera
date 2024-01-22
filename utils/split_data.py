import os
import shutil
from sklearn.model_selection import train_test_split


def split_data(source_folder, dest_folder, train_size=0.7, val_size=0.2, test_size=0.1):
    classes = [
        d
        for d in os.listdir(source_folder)
        if os.path.isdir(os.path.join(source_folder, d))
    ]

    for cls in classes:
        os.makedirs(os.path.join(dest_folder, "train", cls), exist_ok=True)
        os.makedirs(os.path.join(dest_folder, "val", cls), exist_ok=True)
        os.makedirs(os.path.join(dest_folder, "test", cls), exist_ok=True)

        # Get all file names
        all_files = [
            f
            for f in os.listdir(os.path.join(source_folder, cls))
            if os.path.isfile(os.path.join(source_folder, cls, f))
        ]

        # Split data into train and remaining
        train_files, remaining = train_test_split(
            all_files, train_size=train_size, random_state=42
        )
        # Split remaining into val and test
        val_files, test_files = train_test_split(
            remaining, train_size=val_size / (val_size + test_size), random_state=42
        )

        # Copy files to their respective directories
        for f in train_files:
            shutil.copy(
                os.path.join(source_folder, cls, f),
                os.path.join(dest_folder, "train", cls, f),
            )
        for f in val_files:
            shutil.copy(
                os.path.join(source_folder, cls, f),
                os.path.join(dest_folder, "val", cls, f),
            )
        for f in test_files:
            shutil.copy(
                os.path.join(source_folder, cls, f),
                os.path.join(dest_folder, "test", cls, f),
            )


source_folder = "data"
dest_folder = "data_for_learning"
split_data(source_folder, dest_folder)

import os

import cv2
import numpy as np
import pandas as pd

IMG_SIZE = 64  # Image size (64x64 pixels)


def load_images_from_one_folder(folder_path, label=0, class_name="unknown"):
    images = []
    labels = []

    folder_name = os.path.basename(folder_path)

    # Construct the CSV file path
    csv_file_path = get_path_to(f"data/raw/{folder_name}.csv")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Create a dictionary to map file paths to their respective class IDs
    label_dict = dict(zip(df['Path'], df['ClassId']))

    file_names = os.listdir(folder_path)
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):  # Ensure it's a file
            img = cv2.imread(file_path)
            if img is not None:
                # Resize to uniform size
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)

                # Get the label from the dictionary
                rel_path = os.path.relpath(file_path, folder_path)
                label = label_dict.get(f"{folder_name}/{rel_path}", 0) # Default to 0 if path not found
                labels.append(label) # Append the label from CSV
    return np.array(images), np.array(labels), class_name


def load_all_images_from_folder(folder_path):
    images = []
    labels = []
    class_names = os.listdir(folder_path)
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                img = cv2.imread(file_path)
                if img is not None:
                    # Resize to uniform size
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels), class_names


def get_path_to(dir):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../'))
    return os.path.join(project_root, dir)
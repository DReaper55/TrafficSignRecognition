import os

import cv2
import numpy as np

IMG_SIZE = 64  # Image size (64x64 pixels)


def load_images_from_one_folder(folder_path, label=0, class_name="unknown"):
    images = []
    labels = []
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        print(file_path)

        if os.path.isfile(file_path):  # Ensure it's a file
            img = cv2.imread(file_path)
            if img is not None:
                # Resize to uniform size
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)  # Assign the same label to all images
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
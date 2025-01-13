import numpy as np

from src.features.build_features import augment_images
from src.utils.helpers import load_images_from_one_folder, get_path_to, load_all_images_from_folder, normalize_images
from src.visualization.visualize import plot_images


def process_data(images, labels):

    new_images, new_labels = augment_images(images, labels)

    return new_images, new_labels


def get_dataset(is_test=False, is_train=False):
    images = []
    labels = []
    class_names = []

    if is_test:
        test_folder_path = get_path_to('data/raw/Test')
        images, labels, class_names = load_images_from_one_folder(test_folder_path, class_name="Test")


    if is_train:
        train_folder_path = get_path_to('data/raw/Train')
        images, labels, class_names = load_all_images_from_folder(train_folder_path)



    # images = np.concatenate((test_images, train_images), axis=0)
    # labels = np.concatenate((test_labels, train_labels), axis=0)

    # new_images, new_labels = process_data(images, labels)
    #
    # images = np.concatenate((images, new_images), axis=0)
    # labels = np.concatenate((labels, new_labels), axis=0)

    images = normalize_images(images)
    return images, labels


# train_folder_path = get_path_to('data/raw/Train')
# train_images, train_labels, train_class_names = load_all_images_from_folder(train_folder_path)

# images, labels = get_dataset(is_train=True)
#
# print(images.shape)
# print(labels.shape)
#
# plot_images(images, labels)

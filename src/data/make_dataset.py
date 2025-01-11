from keras.src.preprocessing.image import ImageDataGenerator

from src.utils.helpers import load_images_from_one_folder, get_path_to
from src.visualization.visualize import plot_images


def process_data(images, labels, class_names):

    augment_images(images, labels, class_names)


def normalize_images(images):
    # Normalize pixel values to [0, 1]
    return images / 255.0

def augment_images(images, labels, class_names):
    # Data Augmentation
    data_gen = ImageDataGenerator(
        rotation_range=30,        # Rotate images up to 30 degrees
        width_shift_range=0.2,    # Horizontally shift images
        height_shift_range=0.2,   # Vertically shift images
        zoom_range=0.2,           # Zoom in/out
        horizontal_flip=True,     # Flip images horizontally
        brightness_range=[0.8, 1.2],  # Adjust brightness
    )

    # Generate augmented images
    augmented_data = data_gen.flow(images, labels, batch_size=32)

    # Display some augmented images
    augmented_images, augmented_labels = next(augmented_data)

    augmented_images = normalize_images(augmented_images)

    plot_images(augmented_images, augmented_labels, class_names)


folder_path = get_path_to('data/raw/Meta')

images, labels, class_names = load_images_from_one_folder(folder_path, class_name="Meta")

process_data(images, labels, class_names)

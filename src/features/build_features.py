from keras.src.preprocessing.image import ImageDataGenerator

from src.utils.helpers import normalize_images


def augment_images(images, labels):
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

    # augmented_images = normalize_images(augmented_images)

    # plot_images(augmented_images, augmented_labels, class_names)

    return augmented_images, augmented_labels
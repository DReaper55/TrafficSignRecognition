import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from src.data.make_dataset import get_dataset
from src.utils.helpers import get_path_to


def load_and_predict(model_path, new_images):
    # Load the saved model
    model = load_model(model_path)

    # Make predictions
    predictions = model.predict(new_images)
    predicted_classes = np.argmax(predictions, axis=1)

    # Display predictions
    plt.figure(figsize=(15, 5))
    for i in range(min(5, len(new_images))):
        plt.subplot(1, 5, i + 1)
        plt.imshow(new_images[i])
        plt.title(f"Predicted: {[predicted_classes[i]]}")
        plt.axis('off')
    plt.show()

    return predicted_classes


images, labels = get_dataset(is_test=True)

model_save_path = get_path_to(f'models/traffic_sign_model.h5')

pred_classes = load_and_predict(model_save_path, images)

print(f'Predicted classes: {pred_classes}')

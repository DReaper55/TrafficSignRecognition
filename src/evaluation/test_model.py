import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from src.data.make_dataset import get_dataset
from src.utils.helpers import get_path_to


gtsrb_classes = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]


def load_and_predict(new_images, display_graph=False):
    model_path = get_path_to(f'models/traffic_sign_model.h5')

    # Load the saved model
    model = load_model(model_path)

    # Make predictions
    predictions = model.predict(new_images)
    predicted_classes = np.argmax(predictions, axis=1)

    if display_graph:
        # Display predictions
        plt.figure(figsize=(15, 5))
        for i in range(min(5, len(new_images))):
            plt.subplot(1, 5, i + 1)
            plt.imshow(new_images[i])
            plt.title(f"Predicted: {gtsrb_classes[predicted_classes[i]]}")
            plt.axis('off')
        plt.show()

    return predicted_classes


# images, labels = get_dataset(is_test=True)
#
# pred_classes = load_and_predict(images)
#
# print(f'Predicted classes: {pred_classes}')

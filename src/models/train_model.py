from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.data.make_dataset import get_dataset
from src.utils.helpers import get_path_to


def build_cnn(num_conv_layers=3, num_filters=128, activation='relu', dense_units=192, input_shape=(64, 64, 3)):
    model = Sequential()
    # Convolutional and pooling layers
    for _ in range(num_conv_layers):
        model.add(Conv2D(num_filters, (3, 3), activation=activation, padding='same', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    # Fully connected (dense) layers
    model.add(Dense(dense_units, activation=activation))
    model.add(Dropout(0.5))  # Dropout to prevent overfitting
    model.add(Dense(43, activation='softmax'))  # Output layer for 43 classes
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_save_model(X_train, y_train, X_val, y_val, model_file_name='traffic_sign_model.h5'):
    model = build_cnn()

    model_save_path = get_path_to(f'models/{model_file_name}')

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss')

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    return model


X_train, y_train = get_dataset(is_train=True)
X_val, y_val = get_dataset(is_test=True)

train_and_save_model(X_train, y_train, X_val, y_val)

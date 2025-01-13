from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch

from src.data.make_dataset import get_dataset


class CNNHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        # Convolutional layers
        for i in range(hp.Int('num_conv_layers', 1, 3)):
            model.add(Conv2D(filters=hp.Choice('filters', [32, 64, 128]),
                             kernel_size=(3, 3),
                             activation=hp.Choice('activation', ['relu', 'tanh']),
                             padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # Dense layers
        model.add(Dense(units=hp.Int('dense_units', 64, 256, step=64),
                        activation=hp.Choice('activation', ['relu', 'tanh'])))
        model.add(Dropout(0.5))
        model.add(Dense(43, activation='softmax'))  # Output layer for 43 classes
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model


# Initialize the tuner
tuner = RandomSearch(
    CNNHyperModel(),
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='hyperparameter_tuning',
    project_name='traffic_sign_recognition'
)


X_train, y_train = get_dataset(is_train=True)
X_val, y_val = get_dataset(is_test=True)

# Run the tuner
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best Hyperparameters: {best_hyperparameters.values}")

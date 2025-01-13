# Traffic Sign Recognition Project

## Overview
This project focuses on building a deep learning model for traffic sign recognition. The model can classify traffic signs into their respective categories using image data. This application can be integrated into autonomous driving systems to enhance road safety.

## Project Structure
```
project_name/
├── data/
│   ├── raw/                    # Raw data (immutable, original source)
│   ├── processed/              # Processed data (cleaned, transformed)
├── notebooks/                  # Jupyter notebooks for exploration and experimentation
├── src/                        # Source code for the project
│   ├── data/                   # Scripts to download, generate, and process data
│   ├── features/               # Scripts to turn raw data into features for modeling
│   ├── models/                 # Scripts to train models and generate predictions
│   ├── visualization/          # Scripts to create visualizations and plots
│   └── utils/                  # Utility functions and helpers
├── tests/                      # Unit tests
├── models/                     # Serialized models (e.g., saved with joblib or pickle)
├── reports/                    # Generated reports (e.g., for analysis and results)
├── requirements.txt            # Python package dependencies
├── setup.py                    # Installable package configuration
└── README.md                   # Project overview and instructions
```

## Features
- **Data Preprocessing**: Resize, normalize, and augment images to prepare them for training.
- **Model Training**: Build and train a CNN using TensorFlow/Keras.
- **Evaluation**: Evaluate model performance using metrics and visualize training history.
- **Prediction**: Real-time prediction of traffic signs from new images.
- **API Integration**: Expose the model as a REST API for real-time interaction.

## Requirements
- Python 3.8+
- TensorFlow 2.6+
- OpenCV
- NumPy
- Matplotlib
- Flask/FastAPI (for API integration)

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## How to Run

### 1. Data Preparation
- Place raw data in the `data/raw/` directory.
- Use the preprocessing script in `src/data/make_dataset.py` to process and split the data into training and validation sets.

### 2. Model Training
- Use the script in `src/models/train_model.py` to train the CNN model.
- The model will be saved in the `models/` directory.

### 3. Real-Time Prediction
- Use the saved model for prediction using the script in `src/models/predict.py`.
- Example predictions are provided in the notebook under `notebooks/`.

### 4. API Integration
- Start the API server using the script in `src/api/app.py`.
- Test the API using tools like Postman:
    - Endpoint: `http://127.0.0.1:8000/predict/`
    - Method: POST
    - Upload an image file as form data with the key `file`.

### 5. Visualization
- Use `src/visualization/visualize.py` to create plots and visualizations of predictions and model performance.

## Evaluation
The model is evaluated using:
- **Top-1 Accuracy**
- **Confusion Matrix**

## Benchmarks
- **Training Accuracy**: 98.7%
- **Validation Accuracy**: 97.6%
- **Inference Time**: ~25ms per image
- **Model Size**: 21 MB

These benchmarks were achieved using the German Traffic Sign Recognition Benchmark (GTSRB) dataset with a CNN trained on 64x64 images.

## Example
Train the model and evaluate its performance:
```bash
python src/models/train_model.py
```
Predict new data:
```bash
python src/models/predict.py --image path_to_image.jpg
```

## Contributions
Feel free to open issues or submit pull requests for improvements or feature additions.

## License
This project is licensed under the MIT License. See `LICENSE` for more details.


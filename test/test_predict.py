from src.model_pipeline import load_model, predict
import numpy as np
import pytest
import os

def test_predict():
    # Ensure the model file exists
    model_path = "models/gbm_model.joblib"
    assert os.path.exists(model_path), f"Model file not found at {model_path}"

    # Load the model
    model = load_model()

    # Create realistic sample features (replace with actual feature ranges)
    sample_features = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])

    # Make a prediction
    prediction = predict(sample_features)

    # Validate the prediction
    assert prediction in [0, 1], f"Invalid prediction: {prediction}"

    # Optional: Add more assertions to validate the prediction logic
    # For example, check if the prediction matches expected behavior for known inputs
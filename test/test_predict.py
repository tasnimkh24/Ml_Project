import os
import numpy as np
import joblib
from src.model_pipeline import load_model, predict
import pytest

def test_predict():
    # Ensure the model file exists
    model_path = "models/gbm_model.joblib"
    assert os.path.exists(model_path), f"Model file not found at {model_path}"

    try:
        # Load the model
        model = load_model()
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")

    # Create realistic sample features (replace with actual feature ranges)
    sample_features = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])

    try:
        # Make a prediction
        prediction = predict(sample_features)
    except Exception as e:
        pytest.fail(f"Failed to make prediction: {e}")

    # Validate the prediction
    assert prediction in [0, 1], f"Invalid prediction: {prediction}"
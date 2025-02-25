import pytest
import mlflow
import numpy as np
import time
from sklearn.metrics import accuracy_score
from src.model_pipeline import prepare_data, train_model

@pytest.fixture(autouse=True)
def clear_mlflow_state():
    """
    Clear MLflow state before each test.
    """
    mlflow.end_run()  # End any active run
    mlflow.start_run()  # Start a new run
    yield
    mlflow.end_run()  # End the run after the test

def test_train_model():
    """
    Test the model training functionality.
    """
    X_train = np.random.rand(100, 8)  # Données factices
    y_train = np.random.randint(0, 2, 100)  # Labels factices
    
    model = train_model(X_train, y_train)
    assert model is not None  # Vérifiez que le modèle est bien créé

def test_model_accuracy():
    """
    Test the model's accuracy.
    """
    X_train, X_test, y_train, y_test, _, _ = prepare_data("data/train.csv", "data/test.csv")
    
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.8  # Vérifiez que l'accuracy est supérieure à 80%


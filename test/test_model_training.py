import mlflow
from src.model_pipeline import train_model
import numpy as np

def test_train_model():
    X_train = np.random.rand(100, 8)  # Données factices
    y_train = np.random.randint(0, 2, 100)  # Labels factices
    
    # Start an MLflow run for this test
    with mlflow.start_run():
        model = train_model(X_train, y_train)
        assert model is not None  # Vérifiez que le modèle est bien créé
from src.model_pipeline import train_model
import numpy as np

def test_train_model():
    X_train = np.random.rand(100, 8)  # Données factices
    y_train = np.random.randint(0, 2, 100)  # Labels factices
    model = train_model(X_train, y_train)
    assert model is not None

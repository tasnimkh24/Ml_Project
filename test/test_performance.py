import mlflow
import numpy as np
import time
from sklearn.metrics import accuracy_score
from src.model_pipeline import prepare_data, train_model

# Clear any active runs before starting tests
mlflow.end_run()

def test_train_model():
    """
    Test the model training functionality.
    """
    X_train = np.random.rand(100, 8)  # Données factices
    y_train = np.random.randint(0, 2, 100)  # Labels factices
    
    # Start a new MLflow run for this test
    with mlflow.start_run():
        model = train_model(X_train, y_train)
        assert model is not None  # Vérifiez que le modèle est bien créé

def test_model_accuracy():
    """
    Test the model's accuracy.
    """
    X_train, X_test, y_train, y_test, _, _ = prepare_data("data/train.csv", "data/test.csv")
    
    # Start a new MLflow run for this test
    with mlflow.start_run():
        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        assert accuracy > 0.8  # Vérifiez que l'accuracy est supérieure à 80%

def test_training_time():
    """
    Test the training time of the model.
    """
    X_train, _, y_train, _, _, _ = prepare_data("data/train.csv", "data/test.csv")
    
    # Start a new MLflow run for this test
    with mlflow.start_run():
        start_time = time.time()
        train_model(X_train, y_train)
        training_time = time.time() - start_time
        assert training_time < 10, "Training took too long!"
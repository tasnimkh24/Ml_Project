import time
import mlflow
from src.model_pipeline import prepare_data, train_model

def test_training_time():
    X_train, _, y_train, _, _, _ = prepare_data("data/train.csv", "data/test.csv")
    
    # Start an MLflow run for this test
    with mlflow.start_run():
        start_time = time.time()
        train_model(X_train, y_train)
        training_time = time.time() - start_time
        assert training_time < 10, "Training took too long!"
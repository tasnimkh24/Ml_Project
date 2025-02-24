import pytest
from src.model_pipeline import prepare_data

def test_prepare_data():
    X_train, X_test, y_train, y_test, _, _ = prepare_data("data/train.csv", "data/test.csv")
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

from src.model_pipeline import prepare_data, train_model
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    X_train, X_test, y_train, y_test, _, _ = prepare_data("data/train.csv", "data/test.csv")
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.8  # Vérifiez que l'accuracy est supérieure à 80%

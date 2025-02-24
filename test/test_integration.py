from src.model_pipeline import prepare_data, train_model, evaluate_model

def test_pipeline():
    X_train, X_test, y_train, y_test, _, _ = prepare_data("data/train.csv", "data/test.csv")
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)  # Vérifiez qu'il n'y a pas d'erreur

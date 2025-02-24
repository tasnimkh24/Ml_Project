import argparse
import sys
from model_pipeline import prepare_data, train_model, save_model, load_model, evaluate_model, predict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import mlflow
import mlflow.sklearn

def prepare_only(train_path, test_path):
    X_train, X_test, y_train, y_test, X_cluster, y_cluster = prepare_data(train_path, test_path)
    print("\n✅ Data Preparation Completed!")
    print(f"📊 X_train shape: {X_train.shape}")
    print(f"📊 X_test shape: {X_test.shape}")

    # Enregistrer les données préparées dans MLflow
    with mlflow.start_run():
        mlflow.log_artifact(train_path, "data")
        mlflow.log_artifact(test_path, "data")
        print("✅ Données préparées enregistrées dans MLflow.")

def deploy_model():
    # Charger le modèle entraîné
    model = load_model("gbm_model.joblib")
    print("\n🚀 Deploying Model...")

    # Enregistrer le modèle déployé dans MLflow
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "deployed_model")
        print("✅ Model deployed and logged in MLflow!")

def evaluate_only(train_path, test_path):
    # Charger le modèle depuis le dossier models/
    model = load_model("gbm_model.joblib")
    # Préparer les données
    X_train, X_test, y_train, y_test, X_cluster, y_cluster = prepare_data(train_path, test_path)
    print("\n✅ Data Preparation Completed!")
    # Évaluer le modèle
    print("\n📊 Evaluating the model...")
    evaluate_model(model, X_test, y_test)
    print("✅ Model evaluation successful!")

def main(train_path, test_path, prepare_only_flag=False, predict_flag=False, train_flag=False, deploy_flag=False, evaluate_flag=False):
    # Configurer MLflow
    mlflow.set_experiment("Mon_Projet_ML")

    if deploy_flag:
        deploy_model()
    elif prepare_only_flag:
        prepare_only(train_path, test_path)
    elif predict_flag:
        print("\n🎯 Running Prediction Mode...")
        # Charger le modèle
        model = load_model("gbm_model.joblib")  # Nom du fichier correct (relatif à models/)
        # Générer des prédictions
        predictions = predict(model, test_path)
        # Sauvegarder ou afficher les prédictions
        print("✅ Predictions generated successfully!")
        print(predictions)  # Ou sauvegarder dans un fichier
    elif train_flag:
        X_train, X_test, y_train, y_test, X_cluster, y_cluster = prepare_data(train_path, test_path)
        print("\n✅ Data Preparation Completed!")
        print("\n🚀 Training Model...")
        model = train_model(X_train, y_train)  # Cette fonction enregistre le modèle dans MLflow
        save_model(model)  # Sauvegarder le modèle localement (optionnel)
        loaded_model = load_model()  # Charger le modèle pour l'évaluation
        y_pred = loaded_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✅ Model Training Completed! Accuracy: {accuracy:.4f}")
        print("\n📊 Evaluating the model...")
        evaluate_model(model, X_test, y_test)
        print("✅ Model evaluation successful!")
    elif evaluate_flag:
        evaluate_only(train_path, test_path)
    else:
        print("❌ No action specified. Use --prepare, --train, --evaluate, --predict, or --deploy.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the prepare_data function")
    parser.add_argument("--train-data", type=str, required=False, help="Path to the training CSV file")
    parser.add_argument("--test", type=str, required=True, help="Path to the test CSV file")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--prepare", action="store_true", help="Only prepare the data, don't train the model")
    parser.add_argument("--predict", action="store_true", help="Run a prediction using a trained model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--deploy", action="store_true", help="Deploy the model")

    args = parser.parse_args()

    # Vérifier si --train-data est requis
    if (args.train or args.prepare or args.evaluate) and not args.train_data:
        parser.error("❌ --train-data is required for --train, --prepare, or --evaluate.")

    main(args.train_data, args.test, prepare_only_flag=args.prepare, predict_flag=args.predict, train_flag=args.train, deploy_flag=args.deploy, evaluate_flag=args.evaluate)
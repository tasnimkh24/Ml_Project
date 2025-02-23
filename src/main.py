import argparse
import sys
from model_pipeline import prepare_data, train_model, save_model, load_model, evaluate_model, predict
from sklearn.metrics import accuracy_score
import numpy as np

def prepare_only(train_path, test_path):
    X_train, X_test, y_train, y_test, X_cluster, y_cluster = prepare_data(train_path, test_path)
    print("\n✅ Data Preparation Completed!")
    print(f"📊 X_train shape: {X_train.shape}")
    print(f"📊 X_test shape: {X_test.shape}")

def deploy_model():
    # Load the trained model
    model = load_model("gbm_model.joblib")
    print("\n🚀 Deploying Model...")
    # Add deployment logic here (e.g., save to a production environment, deploy to an API, etc.)
    print("✅ Model deployed successfully!")

def evaluate_only(train_path, test_path):
    # Chargez le modèle depuis le dossier models/
    model = load_model("gbm_model.joblib")
    # Préparez les données
    X_train, X_test, y_train, y_test, X_cluster, y_cluster = prepare_data(train_path, test_path)
    print("\n✅ Data Preparation Completed!")
    # Évaluez le modèle
    print("\n📊 Evaluating the model...")
    evaluate_model(model, X_test, y_test)
    print("✅ Model evaluation successful!")
    
def main(train_path, test_path, prepare_only_flag=False, predict_flag=False, train_flag=False, deploy_flag=False, evaluate_flag=False):
    if deploy_flag:
        deploy_model()
    elif prepare_only_flag:
        prepare_only(train_path, test_path)
    elif predict_flag:
        print("\n🎯 Running Prediction Mode...")
        # Load the model
        model = load_model("gbm_model.joblib")  # Correct filename (relative to models/)
        # Generate predictions
        predictions = predict(model, test_path)
        # Save or display predictions
        print("✅ Predictions generated successfully!")
        print(predictions)  # Or save to a file
    elif train_flag:
        X_train, X_test, y_train, y_test, X_cluster, y_cluster = prepare_data(train_path, test_path)
        print("\n✅ Data Preparation Completed!")
        print("\n🚀 Training Model...")
        model = train_model(X_train, y_train)
        save_model(model)
        loaded_model = load_model()
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

    # Check if --train-data is required
    if (args.train or args.prepare or args.evaluate) and not args.train_data:
        parser.error("❌ --train-data is required for --train, --prepare, or --evaluate.")

    main(args.train_data, args.test, prepare_only_flag=args.prepare, predict_flag=args.predict, train_flag=args.train, deploy_flag=args.deploy, evaluate_flag=args.evaluate)
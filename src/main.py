import argparse
from model_pipeline import prepare_data, train_model, save_model, load_model, evaluate_model, deploy_model
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import sys
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set MLflow to use SQLite as the backend store
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Mon_Projet_ML")

def validate_file_path(file_path):
    """
    Validate that the file path exists and is a file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

def prepare_only(train_path, test_path):
    """
    Prepare the data and log it as artifacts in MLflow.
    """
    try:
        X_train, X_test, y_train, y_test, X_cluster, y_cluster = prepare_data(train_path, test_path)
        print("\n✅ Data Preparation Completed!")
        print(f"📊 X_train shape: {X_train.shape}")
        print(f"📊 X_test shape: {X_test.shape}")

        # Log the prepared data as artifacts in MLflow
        with mlflow.start_run():
            mlflow.log_artifact(train_path, "data")
            mlflow.log_artifact(test_path, "data")
            print("✅ Prepared data logged in MLflow.")
    except Exception as e:
        logging.error(f"Error during data preparation: {e}")
        sys.exit(1)

def evaluate_only(train_path, test_path):
    """
    Evaluate the model on the test data.
    """
    try:
        # Load the model from the models/ directory
        model = load_model("models/gbm_model.joblib")
        
        # Prepare the data
        X_train, X_test, y_train, y_test, X_cluster, y_cluster = prepare_data(train_path, test_path)
        print("\n✅ Data Preparation Completed!")
        
        # Evaluate the model
        print("\n📊 Evaluating the model...")
        evaluate_model(model, X_test, y_test)
        print("✅ Model evaluation successful!")
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        sys.exit(1)



def main(train_path, test_path, prepare_only_flag=False, train_flag=False, deploy_flag=False, evaluate_flag=False, stage="Staging"):
    """
    Main function to handle data preparation, training, evaluation, and deployment.
    """
    try:
        # Validate file paths
        validate_file_path(train_path)
        validate_file_path(test_path)

        # Start a single MLflow run for the entire process
        with mlflow.start_run():
            if deploy_flag:
                deploy_model(stage=stage)  # Pass the stage argument
            elif prepare_only_flag:
                prepare_only(train_path, test_path)
            elif train_flag:
                X_train, X_test, y_train, y_test, X_cluster, y_cluster = prepare_data(train_path, test_path)
                print("\n✅ Data Preparation Completed!")
                print("\n🚀 Training Model...")
                model = train_model(X_train, y_train)  # This function logs the model in MLflow
                save_model(model)  # Save the model locally
                loaded_model = load_model()  # Load the model for evaluation
                y_pred = loaded_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"\n✅ Model Training Completed! Accuracy: {accuracy:.4f}")
                print("\n📊 Evaluating the model...")
                evaluate_model(model, X_test, y_test)
                print("✅ Model evaluation successful!")
            elif evaluate_flag:
                evaluate_only(train_path, test_path)
            else:
                print("❌ No action specified. Use --prepare, --train, --evaluate, or --deploy.")
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        sys.exit(1)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the prepare_data function")
    parser.add_argument("--train-data", type=str, required=False, help="Path to the training CSV file")
    parser.add_argument("--test", type=str, required=False, help="Path to the test CSV file")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--prepare", action="store_true", help="Only prepare the data, don't train the model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--deploy", action="store_true", help="Deploy the model")
    parser.add_argument("--stage", type=str, default="Staging", help="Stage to transition the model to (Staging, Production, Archived)")

    args = parser.parse_args()

    # Check if --train-data is required for --train, --prepare, or --evaluate
    if (args.train or args.prepare or args.evaluate) and not args.train_data:
        parser.error("❌ --train-data is required for --train, --prepare, or --evaluate.")

    # Check if --test is required for --prepare, --train, or --evaluate
    if (args.prepare or args.train or args.evaluate) and not args.test:
        parser.error("❌ --test is required for --prepare, --train, or --evaluate.")

    main(args.train_data, args.test, prepare_only_flag=args.prepare, train_flag=args.train, deploy_flag=args.deploy, evaluate_flag=args.evaluate, stage=args.stage)
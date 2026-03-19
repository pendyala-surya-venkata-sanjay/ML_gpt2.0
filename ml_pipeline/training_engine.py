import os
import joblib
from datetime import datetime
import shutil

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

from sklearn.metrics import accuracy_score, r2_score
import sys

class TrainingEngine:

    def __init__(self, X, y, problem_type="classification", feature_engineer=None, preprocessor=None):

        self.X = X
        self.y = y
        self.problem_type = problem_type

        self.feature_engineer = feature_engineer
        self.preprocessor = preprocessor

        self.models = {}
        self.results = {}
        self.trained_models = {}
        self.training_errors = {}

        self.best_model = None
        self.best_model_name = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    # -----------------------------
    # Train Test Split
    # -----------------------------
    def split_data(self):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    # -----------------------------
    # Model Recommendation
    # -----------------------------
    def recommend_models(self):

        if self.problem_type == "classification":

            return {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=42),
                "SVM": SVC(probability=True)
            }

        else:

            return {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=42),
                "SVR": SVR()
            }

    # -----------------------------
    # Evaluate Model
    # -----------------------------
    def evaluate_model(self, y_test, predictions):

        if self.problem_type == "classification":
            return accuracy_score(y_test, predictions)

        else:
            return r2_score(y_test, predictions)

    # -----------------------------
    # Train Models
    # -----------------------------
    def train_models(self):

        for name, model in self.models.items():

            try:

                print(f"Training {name}...")
                
                # Convert to numpy arrays if needed to avoid pandas issues
                X_train = self.X_train.toarray() if hasattr(self.X_train, 'toarray') else self.X_train
                X_test = self.X_test.toarray() if hasattr(self.X_test, 'toarray') else self.X_test
                
                model.fit(X_train, self.y_train)

                predictions = model.predict(X_test)

                score = self.evaluate_model(self.y_test, predictions)

                self.results[name] = score
                self.trained_models[name] = model

                print(f"{name} trained successfully. Score: {score}")

            except Exception as e:

                error_msg = f"Training failed for {name}: {str(e)}"
                print(error_msg)
                import traceback
                full_error = traceback.format_exc()
                print(f"Full error details: {full_error}")
                self.training_errors[name] = str(e)

    # -----------------------------
    # Select Best Model
    # -----------------------------
    def select_best_model(self):

        if not self.results:
            raise ValueError("No models were trained successfully.")

        self.best_model_name = max(self.results, key=self.results.get)
        self.best_model = self.trained_models[self.best_model_name]

        print(f"Best Model Selected: {self.best_model_name}")

    # -----------------------------
    # Save Pipeline
    # -----------------------------
    def save_pipeline(self, output_path=None):

        if not os.path.exists("models"):
            os.makedirs("models")

        pipeline = {
            "feature_engineer": self.feature_engineer,
            "preprocessor": self.preprocessor,
            "model": self.best_model,
            "model_name": self.best_model_name
        }

        if output_path is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            file_path = f"models/ml_pipeline_{ts}.pkl"
        else:
            file_path = output_path

        joblib.dump(pipeline, file_path)

        # Keep a stable "latest" path for backwards compatibility.
        try:
            shutil.copy(file_path, "models/ml_pipeline.pkl")
        except Exception:
            pass

        print("Pipeline saved at:", file_path)

        return file_path

    # -----------------------------
    # Main Training Function
    # -----------------------------
    def train(self):

        print("Starting training pipeline...")

        # Step 1: Split data
        self.split_data()

        # Step 2: Recommend models
        self.models = self.recommend_models()

        # Step 3: Train models
        self.train_models()

        # Step 4: Select best model
        self.select_best_model()

        # Step 5: Save pipeline
        pipeline_path = self.save_pipeline()

        return {
            "best_model": self.best_model_name,
            "results": self.results,
            "pipeline_path": pipeline_path,
            "training_errors": self.training_errors
        }
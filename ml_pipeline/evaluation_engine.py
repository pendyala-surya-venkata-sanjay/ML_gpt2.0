from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


class EvaluationEngine:

    def __init__(self, model, X_test, y_test, problem_type):

        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.problem_type = problem_type

    # -----------------------------
    # Classification Metrics
    # -----------------------------
    def evaluate_classification(self):

        predictions = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, predictions)

        precision = precision_score(
            self.y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            self.y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            self.y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        cm = confusion_matrix(self.y_test, predictions)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist()
        }

    # -----------------------------
    # Regression Metrics
    # -----------------------------
    def evaluate_regression(self):

        predictions = self.model.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, predictions)

        mse = mean_squared_error(self.y_test, predictions)

        rmse = np.sqrt(mse)

        r2 = r2_score(self.y_test, predictions)

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2_score": float(r2)
        }

    # -----------------------------
    # Smart Evaluation
    # -----------------------------
    def evaluate(self):

        if self.problem_type == "classification":

            return {
                "problem_type": "classification",
                "metrics": self.evaluate_classification()
            }

        else:

            return {
                "problem_type": "regression",
                "metrics": self.evaluate_regression()
            }
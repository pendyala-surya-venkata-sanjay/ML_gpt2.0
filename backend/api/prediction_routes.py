from fastapi import APIRouter
import joblib
import pandas as pd
import os
import numpy as np
from functools import lru_cache

from backend.services.logger import logger
from backend.services import model_registry

router = APIRouter()


@lru_cache(maxsize=4)
def _load_pipeline(model_path: str):
    return joblib.load(model_path)


@router.post("/predict")
def predict(input_data: dict, model_id: str = None):

    try:

        # If model_id is provided (or an active model is set), load that model.
        selected_model_id = model_id or model_registry.get_active_model_id()
        model_path = "models/ml_pipeline.pkl"

        if selected_model_id:
            m = model_registry.get_model_by_id(selected_model_id)
            if m and m.get("pipeline_path"):
                model_path = m["pipeline_path"]

        if not os.path.exists(model_path):

            logger.error("Prediction attempted without trained model")

            return {
                "status": "error",
                "message": "Model not trained yet"
            }

        pipeline = _load_pipeline(model_path)

        feature_engineer = pipeline["feature_engineer"]
        preprocessor = pipeline["preprocessor"]
        model = pipeline["model"]

        df = pd.DataFrame([input_data])

        # Feature engineering
        feature_engineer.df = df
        df = feature_engineer.apply_all()

        # Align input columns to what the trained ColumnTransformer expects.
        # This prevents KeyError/missing-column failures when the user omits some features.
        try:
            column_transformer = preprocessor.preprocessor

            numeric_expected: set[str] = set()
            categorical_expected: set[str] = set()
            expected_cols: set[str] = set()

            for name, _trans, cols in (getattr(column_transformer, "transformers", None) or []):
                if cols is None:
                    continue
                # `cols` is sometimes a numpy array / list of column names.
                # Only treat special values like "drop"/"passthrough" when they are strings.
                if isinstance(cols, str) and cols in {"drop", "passthrough"}:
                    continue

                cols_list: list[str] = []
                if isinstance(cols, (list, tuple, pd.Index)):
                    cols_list = list(cols)
                elif isinstance(cols, str):
                    cols_list = [cols]
                else:
                    try:
                        cols_list = list(cols)  # best-effort
                    except Exception:
                        cols_list = []

                for c in cols_list:
                    expected_cols.add(str(c))
                    if str(name).lower().startswith("num"):
                        numeric_expected.add(str(c))
                    elif str(name).lower().startswith("cat"):
                        categorical_expected.add(str(c))

            # Add any missing columns as NaN so imputers can handle them.
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = np.nan

            # Ensure numeric columns are numeric (strings -> numbers when possible).
            for col in numeric_expected:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception as e:
            # Best-effort alignment; prediction should still try, but we want visibility.
            logger.error(f"Prediction input alignment failed: {str(e)}")

        # Preprocessing
        X = preprocessor.preprocessor.transform(df)

        prediction = model.predict(X)[0]
        # Ensure prediction is JSON-serializable (FastAPI may not handle numpy scalars well).
        if hasattr(prediction, "item"):
            prediction = prediction.item()

        response = {
            "status": "success",
            "prediction": prediction
        }

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X)

            confidence = float(np.max(probabilities))

            response["confidence"] = confidence

        logger.info(f"Prediction successful: {prediction}")

        if selected_model_id:
            response["model_id"] = selected_model_id

        return response

    except Exception as e:

        logger.error(f"Prediction error: {str(e)}")

        return {
            "status": "error",
            "message": str(e)
        }
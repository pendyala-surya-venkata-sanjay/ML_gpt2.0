from fastapi import APIRouter
import joblib
import pandas as pd
import os
import numpy as np

from backend.services.logger import logger
from backend.services import model_registry

router = APIRouter()


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

        pipeline = joblib.load(model_path)

        feature_engineer = pipeline["feature_engineer"]
        preprocessor = pipeline["preprocessor"]
        model = pipeline["model"]

        df = pd.DataFrame([input_data])

        # Feature engineering
        feature_engineer.df = df
        df = feature_engineer.apply_all()

        # Preprocessing
        X = preprocessor.preprocessor.transform(df)

        prediction = model.predict(X)[0]

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
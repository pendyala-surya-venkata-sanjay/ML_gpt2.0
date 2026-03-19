import joblib
import pandas as pd
import os


class PredictionEngine:

    def __init__(self, model_path):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        pipeline = joblib.load(model_path)

        self.feature_engineer = pipeline["feature_engineer"]
        self.preprocessor = pipeline["preprocessor"]
        self.model = pipeline["model"]

    def predict_single(self, input_data):

        df = pd.DataFrame([input_data])

        self.feature_engineer.df = df
        df = self.feature_engineer.apply_all()

        X = self.preprocessor.preprocessor.transform(df)

        prediction = self.model.predict(X)[0]

        try:
            prediction = prediction.item()
        except:
            pass

        return prediction
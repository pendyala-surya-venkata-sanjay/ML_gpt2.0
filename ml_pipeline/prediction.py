import joblib
import pandas as pd
import os
import numpy as np
import re
from functools import lru_cache


class PredictionEngine:

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_pipeline(model_path: str):
        return joblib.load(model_path)

    def __init__(self, model_path):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        pipeline = self._load_pipeline(model_path)

        self.feature_engineer = pipeline["feature_engineer"]
        self.preprocessor = pipeline["preprocessor"]
        self.model = pipeline["model"]

        self._expected_numeric_cols, self._expected_categorical_cols = self._extract_expected_feature_columns()

    def _extract_expected_feature_columns(self):
        """
        Extract expected input feature columns from the fitted sklearn ColumnTransformer.
        Used to safely align user input (including empty input) to what the model expects.
        """
        ct = getattr(self.preprocessor, "preprocessor", None)
        if ct is None:
            return [], []

        numeric_cols: list[str] = []
        categorical_cols: list[str] = []

        transformers = getattr(ct, "transformers", None) or []
        for name, _trans, cols in transformers:
            if cols is None:
                continue
            if isinstance(cols, str) and cols in {"drop", "passthrough"}:
                continue

            if isinstance(cols, (list, tuple, pd.Index)):
                cols_list = list(cols)
            elif isinstance(cols, str):
                cols_list = [cols]
            else:
                try:
                    cols_list = list(cols)
                except Exception:
                    cols_list = []

            if str(name).lower().startswith("num"):
                for c in cols_list:
                    cs = str(c)
                    if cs not in numeric_cols:
                        numeric_cols.append(cs)
            elif str(name).lower().startswith("cat"):
                for c in cols_list:
                    cs = str(c)
                    if cs not in categorical_cols:
                        categorical_cols.append(cs)

        return numeric_cols, categorical_cols

    def get_expected_feature_columns(self):
        """Return expected numeric and categorical columns (in stable order)."""
        return self._expected_numeric_cols, self._expected_categorical_cols

    def get_expected_raw_numeric_feature_columns(self):
        """
        Return expected numeric *input* columns (before interaction feature creation).
        Our feature engineering creates pairwise interaction columns with suffix `_interaction`.
        When users provide a simple numeric list, it typically corresponds to the raw base
        numeric inputs, not the engineered interaction columns.
        """
        return [c for c in self._expected_numeric_cols if not str(c).endswith("_interaction")]

    def predict_single(self, input_data):
        # Build a DataFrame even for empty/unexpected inputs.
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = pd.DataFrame([{}])

        self.feature_engineer.df = df
        df = self.feature_engineer.apply_all()

        # Align to the trained ColumnTransformer so missing/empty user input
        # never causes sklearn to crash (e.g. column_names becomes None).
        try:
            expected_cols = list(dict.fromkeys(self._expected_numeric_cols + self._expected_categorical_cols))
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = np.nan

            for col in self._expected_numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass

        X = self.preprocessor.preprocessor.transform(df)

        prediction = self.model.predict(X)[0]

        try:
            prediction = prediction.item()
        except:
            pass

        return prediction
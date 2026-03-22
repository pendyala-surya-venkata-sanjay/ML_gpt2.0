import pandas as pd


class FeatureEngineering:

    def __init__(self, df):
        self.df = df.copy()

    def get_features(self):
        """Return the engineered feature dataframe."""
        return self.df

    def interaction_features(self):

        # Treat any numeric dtype as numeric.
        numeric_cols = self.df.select_dtypes(include=["number"]).columns

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):

                col1 = numeric_cols[i]
                col2 = numeric_cols[j]

                new_col = f"{col1}_{col2}_interaction"

                self.df[new_col] = self.df[col1] * self.df[col2]

        return self.df

    def datetime_features(self):

        for col in self.df.columns:

            if "date" in col.lower():

                # Be tolerant to unexpected formats coming from new datasets.
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

                self.df[f"{col}_year"] = self.df[col].dt.year
                self.df[f"{col}_month"] = self.df[col].dt.month
                self.df[f"{col}_day"] = self.df[col].dt.day

        return self.df

    def apply_all(self):

        self.datetime_features()

        self.interaction_features()

        return self.df

    def apply_intelligent_engineering(self, analysis_result=None):
        """
        Backwards-compatible, safer feature engineering.
        - Always handles datetime-like columns.
        - Adds interaction features only when there are at least two numeric columns.
        - Ignores analysis_result for now but keeps it for future adaptive logic.
        """
        # Always extract datetime features; this is cheap and robust.
        self.datetime_features()

        # Only create interaction terms when it makes sense.
        numeric_cols = self.df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) >= 2:
            self.interaction_features()

        return self.df
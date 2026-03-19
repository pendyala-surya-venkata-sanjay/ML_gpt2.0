import pandas as pd


class FeatureEngineering:

    def __init__(self, df):
        self.df = df.copy()

    def interaction_features(self):

        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

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

                self.df[col] = pd.to_datetime(self.df[col])

                self.df[f"{col}_year"] = self.df[col].dt.year
                self.df[f"{col}_month"] = self.df[col].dt.month
                self.df[f"{col}_day"] = self.df[col].dt.day

        return self.df

    def apply_all(self):

        self.datetime_features()

        self.interaction_features()

        return self.df
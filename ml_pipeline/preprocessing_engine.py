import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class PreprocessingEngine:

    def __init__(self, data, target_column):

        self.data = data
        self.target_column = target_column

        self.X = None
        self.y = None

        self.preprocessor = None

    def preprocess(self):

        # Separate features and target
        self.X = self.data.drop(columns=[self.target_column])
        
        # Remove ID-like columns that shouldn't be used for training
        id_columns = [col for col in self.X.columns if col.lower() in ['id', 'identifier', 'index', 'row_id', 'sample_id']]
        if id_columns:
            self.X = self.X.drop(columns=id_columns)
            print(f"Removed ID columns: {id_columns}")

        self.y = self.data[self.target_column]
        
        # Remove rows where target is NaN
        if self.y.isnull().any():
            print(f"Found {self.y.isnull().sum()} rows with NaN in target column. Removing them...")
            valid_indices = ~self.y.isnull()
            self.X = self.X[valid_indices]
            self.y = self.y[valid_indices]
            print(f"Remaining rows after cleaning: {len(self.y)}")

        # Guard: after cleaning, there must be at least one row.
        if self.y is None or len(self.y) == 0:
            raise ValueError("No valid rows available after cleaning the target column (all rows removed).")

        # Detect column types
        # Treat any numeric dtype as numeric.
        numeric_features = self.X.select_dtypes(include=["number"]).columns
        categorical_features = self.X.select_dtypes(include=["object"]).columns

        # Guard: avoid creating an empty ColumnTransformer (can crash downstream estimators).
        if len(numeric_features) == 0 and len(categorical_features) == 0:
            raise ValueError("No usable feature columns found. Ensure your dataset has numeric or object columns.")

        # Numeric pipeline
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]
        )

        # Categorical pipeline
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        # Combine both pipelines
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features)
            ]
        )

        X_processed = self.preprocessor.fit_transform(self.X)

        return {
            "X": X_processed,
            "y": self.y,
            "preprocessor": self.preprocessor
        }
    
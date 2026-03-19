import pandas as pd
import numpy as np


class DatasetAnalyzer:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

    # ------------------------------------------------
    # Detect target column
    # ------------------------------------------------
    def detect_target_column(self):

        possible_targets = ["target", "label", "class", "output", "outcome", "diagnosis"]
        
        # First check for common target column names
        for col in self.df.columns:
            if col.lower() in possible_targets:
                return col
        
        # If no common targets found, look for columns with few unique values (likely categorical targets)
        for col in self.df.columns:
            if col.lower() not in ['id', 'identifier', 'index', 'row_id', 'sample_id', 'unnamed']:
                unique_count = self.df[col].nunique()
                if unique_count <= 10 and unique_count > 1:  # Between 2-10 unique values
                    return col
        
        # Last resort: return the last non-ID column
        non_id_cols = [col for col in self.df.columns if 'id' not in col.lower() and 'unnamed' not in col.lower()]
        if non_id_cols:
            return non_id_cols[-1]
        
        return self.df.columns[-1]

    # ------------------------------------------------
    # Detect problem type
    # ------------------------------------------------
    def detect_problem_type(self, target_column):

        target = self.df[target_column]

        unique_values = target.nunique()
        total_rows = len(target)

        if target.dtype == "object":
            return "classification"

        if unique_values <= 10:
            return "classification"

        if unique_values > total_rows * 0.1:
            return "regression"

        return "classification"

    # ------------------------------------------------
    # Main analysis function
    # ------------------------------------------------
    def analyze(self):

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()

        # Missing values
        missing_values = self.df.isnull().sum()
        missing_values = missing_values[missing_values > 0].to_dict()

        # Duplicate rows
        duplicates = int(self.df.duplicated().sum())

        # Constant columns
        constant_columns = [
            col for col in self.df.columns
            if self.df[col].nunique() == 1
        ]

        # Target detection
        target_column = self.detect_target_column()

        # Problem type
        problem_type = self.detect_problem_type(target_column)

        # Class distribution
        class_distribution = {}
        imbalance_detected = False

        if problem_type == "classification":

            class_distribution = (
                self.df[target_column]
                .value_counts(normalize=True)
                .to_dict()
            )

            if len(class_distribution) > 0 and max(class_distribution.values()) > 0.8:
                imbalance_detected = True

        # Correlation detection
        high_correlation_pairs = []

        if len(numeric_cols) > 1:

            corr_matrix = self.df[numeric_cols].corr()

            for i in range(len(corr_matrix.columns)):
                for j in range(i):

                    corr_value = corr_matrix.iloc[i, j]

                    if abs(corr_value) > 0.8:

                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]

                        high_correlation_pairs.append({
                            "feature_1": col1,
                            "feature_2": col2,
                            "correlation": float(corr_value)
                        })

        # ------------------------------------------------
        # Dataset preview
        # ------------------------------------------------
        preview = self.df.head(5).to_dict(orient="records")

        # ------------------------------------------------
        # Feature type summary
        # ------------------------------------------------
        feature_summary = {
            "numeric_feature_count": len(numeric_cols),
            "categorical_feature_count": len(categorical_cols)
        }

        # ------------------------------------------------
        # Target statistics
        # ------------------------------------------------
        target_statistics = {}

        if target_column in numeric_cols:

            target_statistics = {
                "mean": float(self.df[target_column].mean()),
                "std": float(self.df[target_column].std()),
                "min": float(self.df[target_column].min()),
                "max": float(self.df[target_column].max())
            }

        # ------------------------------------------------
        # Final report
        # ------------------------------------------------
        result = {

            "rows": self.df.shape[0],
            "columns": self.df.shape[1],

            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,

            "missing_values": missing_values,
            "duplicate_rows": duplicates,

            "constant_columns": constant_columns,

            "target_column": target_column,
            "problem_type": problem_type,

            "class_distribution": class_distribution,
            "class_imbalance_detected": imbalance_detected,

            "high_correlation_pairs": high_correlation_pairs,

            "feature_summary": feature_summary,

            "target_statistics": target_statistics,

            "dataset_preview": preview
        }

        return result
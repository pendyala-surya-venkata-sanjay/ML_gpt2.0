import pandas as pd


class DatasetTools:

    def __init__(self, dataset_path):

        self.df = pd.read_csv(dataset_path)

    def missing_values(self):

        return self.df.isnull().sum().to_dict()

    def dataset_summary(self):

        return {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "column_names": list(self.df.columns)
        }

    def feature_types(self):

        return self.df.dtypes.astype(str).to_dict()

    def correlation_matrix(self):

        return self.df.corr(numeric_only=True).to_dict()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")


class VisualizationEngine:

    def __init__(self, df, target_column=None):

        self.df = df
        self.target_column = target_column

        os.makedirs("visualizations", exist_ok=True)

    # -----------------------------
    # Correlation Heatmap
    # -----------------------------
    def correlation_heatmap(self):

        numeric_df = self.df.select_dtypes(include=['int64', 'float64'])

        plt.figure(figsize=(8,6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")

        file_path = "visualizations/correlation_heatmap.png"
        plt.savefig(file_path)
        plt.close()

        return file_path

    # -----------------------------
    # Feature Distributions
    # -----------------------------
    def feature_distributions(self):

        numeric_cols = self.df.select_dtypes(include=['int64','float64']).columns

        paths = []

        for col in numeric_cols:

            plt.figure()

            sns.histplot(self.df[col], kde=True)

            file_path = f"visualizations/{col}_distribution.png"

            plt.savefig(file_path)

            plt.close()

            paths.append(file_path)

        return paths

    # -----------------------------
    # Missing Values Chart
    # -----------------------------
    def missing_values_chart(self):

        missing = self.df.isnull().sum()

        missing = missing[missing > 0]

        if missing.empty:
            return None

        plt.figure()

        missing.plot(kind="bar")

        plt.title("Missing Values")

        file_path = "visualizations/missing_values.png"

        plt.savefig(file_path)

        plt.close()

        return file_path

    # -----------------------------
    # Target Distribution
    # -----------------------------
    def target_distribution(self):

        if self.target_column is None:
            return None

        plt.figure()

        sns.countplot(x=self.df[self.target_column])

        file_path = "visualizations/target_distribution.png"

        plt.savefig(file_path)

        plt.close()

        return file_path

    # -----------------------------
    # Generate All Visualizations
    # -----------------------------
    def generate_all(self):

        results = {}

        results["correlation_heatmap"] = self.correlation_heatmap()

        results["feature_distributions"] = self.feature_distributions()

        results["missing_values_chart"] = self.missing_values_chart()

        results["target_distribution"] = self.target_distribution()

        return results
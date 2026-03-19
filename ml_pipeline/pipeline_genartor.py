import os


class PipelineGenerator:

    def __init__(self, dataset_path, target_column, problem_type, best_model):

        self.dataset_path = dataset_path
        self.target_column = target_column
        self.problem_type = problem_type
        self.best_model = best_model

    # --------------------------------
    # Generate Model Import
    # --------------------------------
    def get_model_import(self):

        imports = {
            "Logistic Regression": "from sklearn.linear_model import LogisticRegression",
            "Random Forest": "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor",
            "SVM": "from sklearn.svm import SVC, SVR",
            "Gradient Boosting": "from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor",
            "Linear Regression": "from sklearn.linear_model import LinearRegression"
        }

        return imports.get(self.best_model, "")

    # --------------------------------
    # Generate Model Initialization
    # --------------------------------
    def get_model_initialization(self):

        mapping = {
            "Logistic Regression": "model = LogisticRegression()",
            "Random Forest": "model = RandomForestClassifier()" if self.problem_type == "classification" else "model = RandomForestRegressor()",
            "SVM": "model = SVC()" if self.problem_type == "classification" else "model = SVR()",
            "Gradient Boosting": "model = GradientBoostingClassifier()" if self.problem_type == "classification" else "model = GradientBoostingRegressor()",
            "Linear Regression": "model = LinearRegression()"
        }

        return mapping.get(self.best_model, "")

    # --------------------------------
    # Generate Pipeline Code
    # --------------------------------
    def generate_code(self):

        model_import = self.get_model_import()

        model_init = self.get_model_initialization()

        code = f"""
# -----------------------------------
# Auto Generated ML Pipeline
# -----------------------------------

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

{model_import}

# -----------------------------------
# Load Dataset
# -----------------------------------

df = pd.read_csv("{self.dataset_path}")

# -----------------------------------
# Split Features and Target
# -----------------------------------

X = df.drop("{self.target_column}", axis=1)

y = df["{self.target_column}"]

# -----------------------------------
# Encode Categorical Columns
# -----------------------------------

for col in X.select_dtypes(include="object").columns:

    le = LabelEncoder()

    X[col] = le.fit_transform(X[col])

# -----------------------------------
# Train Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# Feature Scaling
# -----------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# -----------------------------------
# Model Training
# -----------------------------------

{model_init}

model.fit(X_train, y_train)

# -----------------------------------
# Prediction
# -----------------------------------

predictions = model.predict(X_test)

print("Predictions:")

print(predictions)

"""

        return code

    # --------------------------------
    # Save Pipeline Code
    # --------------------------------
    def save_pipeline(self):

        code = self.generate_code()

        os.makedirs("generated_pipelines", exist_ok=True)

        file_path = "generated_pipelines/generated_pipeline.py"

        with open(file_path, "w") as f:
            f.write(code)

        return file_path
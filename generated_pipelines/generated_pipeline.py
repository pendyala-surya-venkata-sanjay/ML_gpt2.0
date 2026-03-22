
# -----------------------------------
# Auto Generated ML Pipeline
# -----------------------------------

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

# -----------------------------------
# Load Dataset
# -----------------------------------

df = pd.read_csv("C:/Users/hp/Desktop/projects/ml_assistant/datasets/uploaded_datasets/crop_yeild_dataset1.csv")

# -----------------------------------
# Split Features and Target
# -----------------------------------

X = df.drop("State", axis=1)

y = df["State"]

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

model = LogisticRegression()

model.fit(X_train, y_train)

# -----------------------------------
# Prediction
# -----------------------------------

predictions = model.predict(X_test)

print("Predictions:")

print(predictions)


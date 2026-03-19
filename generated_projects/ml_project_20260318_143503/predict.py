
import joblib
import pandas as pd

model = joblib.load("model.pkl")

data = pd.DataFrame([{
    "age": 32,
    "salary": 70000,
    "experience": 6,
    "city": "Delhi"
}])

prediction = model.predict(data)

print("Prediction:", prediction)

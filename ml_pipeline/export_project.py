import os
import zipfile
import shutil
from datetime import datetime


class ProjectExporter:

    def __init__(self):

        self.export_folder = "generated_projects"
        self.model_path = "models/ml_pipeline.pkl"

    def export_project(self, model_path=None, project_name=None):

        if not os.path.exists(self.export_folder):
            os.makedirs(self.export_folder)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        project_name = project_name or f"ml_project_{ts}"
        project_path = os.path.join(self.export_folder, project_name)

        if os.path.exists(project_path):
            shutil.rmtree(project_path)

        os.makedirs(project_path)

        # copy model
        src_model = model_path or self.model_path
        shutil.copy(src_model, os.path.join(project_path, "model.pkl"))

        # create prediction script
        prediction_code = """
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
"""

        with open(os.path.join(project_path, "predict.py"), "w") as f:
            f.write(prediction_code)

        # zip project
        zip_path = os.path.join(self.export_folder, f"{project_name}.zip")

        with zipfile.ZipFile(zip_path, "w") as zipf:

            for root, dirs, files in os.walk(project_path):

                for file in files:

                    full_path = os.path.join(root, file)

                    zipf.write(
                        full_path,
                        os.path.relpath(full_path, project_path)
                    )

        return zip_path
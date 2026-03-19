from backend.tools.dataset_tools import DatasetTools
from ml_pipeline.ml_pipeline_controller import MLPipelineController
from ml_pipeline.prediction import PredictionEngine


class ToolRouter:

    def run_tool(self, intent, dataset_path=None, message=None):

        # -------------------------
        # TRAIN MODEL
        # -------------------------
        if intent == "train_model":

            pipeline = MLPipelineController(dataset_path)

            result = pipeline.run_pipeline()

            return "\n".join(result["logs"])

        # -------------------------
        # PREDICTION
        # -------------------------
        if intent == "prediction":

            model_path = "models/ml_pipeline.pkl"

            predictor = PredictionEngine(model_path)

            data = self.parse_prediction_input(message)

            prediction = predictor.predict_single(data)

            return f"Prediction: {prediction}"

        # -------------------------
        # DATASET QUESTIONS
        # -------------------------
        if intent == "dataset_question":

            tools = DatasetTools(dataset_path)

            message = message.lower()

            if "missing" in message or "null" in message:
                return tools.missing_values()

            if "summary" in message or "dataset" in message:
                return tools.dataset_summary()

            if "rows" in message:
                return {"rows": tools.dataset_summary()["rows"]}

            if "columns" in message:
                return {"columns": tools.dataset_summary()["columns"]}

        # -------------------------
        # VISUALIZATION
        # -------------------------
        if intent == "visualization":

            return "Visualization feature will generate plots from the dataset."

        return "Tool execution failed"

    # -------------------------
    # PARSE PREDICTION INPUT
    # -------------------------
    def parse_prediction_input(self, message):

        data = {}

        parts = message.split()

        for part in parts:

            if "=" in part:

                key, value = part.split("=")

                try:
                    value = float(value)
                except:
                    pass

                data[key] = value

        return data
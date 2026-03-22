from backend.tools.dataset_tools import DatasetTools
from ml_pipeline.ml_pipeline_controller import MLPipelineController
from ml_pipeline.prediction import PredictionEngine
import re
from backend.services import model_registry


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

            # Use the currently active model (trained via /train_model),
            # not just the latest `models/ml_pipeline.pkl` artifact.
            selected_model_id = model_registry.get_active_model_id()
            model_path = "models/ml_pipeline.pkl"
            if selected_model_id:
                m = model_registry.get_model_by_id(selected_model_id)
                pipeline_path = m.get("pipeline_path") if m else None
                if pipeline_path:
                    model_path = pipeline_path

            predictor = PredictionEngine(model_path)

            # For chat-style numeric list input, map values to the raw numeric inputs
            # (excluding engineered `_interaction` features).
            expected_numeric_cols = predictor.get_expected_raw_numeric_feature_columns()
            data = self.parse_prediction_input(message, expected_numeric_cols=expected_numeric_cols)

            try:
                prediction = predictor.predict_single(data)
                return f"Prediction: {prediction}"
            except Exception as e:
                return f"Prediction failed: {str(e)}"

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
    def parse_prediction_input(self, message, expected_numeric_cols=None):

        data = {}

        if not message:
            return data

        parts = str(message).split()

        for part in parts:

            if "=" in part:

                key, value = part.split("=")

                try:
                    value = float(value)
                except:
                    pass

                data[key] = value

        # If no key=value pairs were provided, try parsing comma/space-separated numbers
        # and map them onto expected numeric feature columns in order.
        if data:
            return data

        if expected_numeric_cols is None:
            return data

        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", str(message))
        values: list[float] = []
        for n in nums:
            try:
                values.append(float(n))
            except Exception:
                continue

        for i, v in enumerate(values):
            if i >= len(expected_numeric_cols):
                break
            data[str(expected_numeric_cols[i])] = v

        return data
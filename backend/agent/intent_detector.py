class IntentDetector:

    def detect_intent(self, message):

        msg = message.lower()

        if "train" in msg:
            return "train_model"

        if "predict" in msg:
            return "prediction"

        if "missing" in msg or "null" in msg:
            return "dataset_question"

        if "dataset" in msg or "summary" in msg:
            return "dataset_question"

        if "rows" in msg or "columns" in msg:
            return "dataset_question"

        if "plot" in msg or "graph" in msg:
            return "visualization"

        return "llm_question"
import json
import os


class KnowledgeService:

    def __init__(self):

        self.concepts_file = "knowledge_base/ml_concepts.json"
        self.models_file = "knowledge_base/model_explanations.json"
        self.preprocessing_file = "knowledge_base/preprocessing_guides.json"
        self.training_file = "knowledge_base/training_guides.json"

    def normalize_text(self, text):

        text = text.lower()
        text = text.replace("_", "")
        text = text.replace(" ", "")

        return text
    def load_json(self, file_path):

        if not os.path.exists(file_path):
            return {}

        with open(file_path, "r") as f:
            return json.load(f)


    def get_concept(self, message):

        normalized_message = self.normalize_text(message)

        files = [
            self.concepts_file,
            self.models_file,
            self.preprocessing_file,
            self.training_file
        ]

        for file in files:

            data = self.load_json(file)

            for key in data:

                normalized_key = self.normalize_text(key)

                if normalized_key in normalized_message or normalized_message in normalized_key:
                    return data[key]

        return None


    def add_new_concept(self, question, answer):

        question = question.lower()

        data = self.load_json(self.concepts_file)

        data[question] = answer

        with open(self.concepts_file, "w") as f:
            json.dump(data, f, indent=4)

        return True
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------
# Singleton Model Loader
# -------------------------------------------------

_model = None


def get_model():

    global _model

    if _model is None:

        print("Loading semantic model...")

        _model = SentenceTransformer("all-MiniLM-L6-v2")

    return _model


# -------------------------------------------------
# Semantic Search Engine
# -------------------------------------------------

class SemanticSearch:

    def __init__(self):

        self.model = get_model()

        self.documents = []
        self.texts = []

        self.load_knowledge()

        if len(self.texts) > 0:

            self.embeddings = self.model.encode(self.texts)

        else:

            self.embeddings = []

    def load_knowledge(self):

        base_path = "knowledge_base"

        for file in os.listdir(base_path):

            if file.endswith(".json"):

                path = os.path.join(base_path, file)

                with open(path, "r") as f:

                    data = json.load(f)

                    for key, value in data.items():

                        text = key.replace("_", " ") + " " + value

                        self.documents.append({
                            "topic": key,
                            "text": value
                        })

                        self.texts.append(text)

    def search(self, query):

        if len(self.texts) == 0:
            return None

        query = query.replace("_", " ")

        query_embedding = self.model.encode([query])

        scores = cosine_similarity(query_embedding, self.embeddings)[0]

        best_index = np.argmax(scores)

        return self.documents[best_index]
class IntentParser:

    def parse(self, user_message):

        message = user_message.lower()

        # training
        if "train" in message:
            return "train_model"

        # prediction
        if "predict" in message:
            return "predict"

        # dataset upload
        if "upload dataset" in message:
            return "upload_dataset"

        # concept explanation
        if (
            "explain" in message
            or "what is" in message
            or "tell me about" in message
            or "how does" in message
        ):
            return "explain_concept"

        # ML keywords (allow short queries)
        ml_terms = [
            "random forest",
            "svm",
            "logistic regression",
            "decision tree",
            "pca",
            "lda",
            "kmeans",
            "gradient boosting",
            "naive bayes",
            "knn"
        ]

        for term in ml_terms:
            if term in message:
                return "explain_concept"

        if "show visualization" in message:
            return "show_visualization"

        if "compare models" in message:
            return "compare_models"

        if "recommend model" in message or "suggest model" in message:
            return "recommend_model"

        return "general_question"
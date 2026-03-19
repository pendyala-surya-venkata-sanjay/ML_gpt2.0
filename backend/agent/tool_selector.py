from backend.services.llm_service import LLMService


class ToolSelector:

    def __init__(self):

        self.llm = LLMService()

        self.tools = [
            "train_model",
            "prediction",
            "dataset_question",
            "ml_question"
        ]

    def select_tool(self, message):

        prompt = f"""
You are an AI agent responsible for selecting the correct tool.

TOOLS AVAILABLE:

train_model → when user wants to train a machine learning model.

prediction → when user wants to make prediction using trained model.

dataset_question → when user asks about dataset statistics, missing values, correlations, etc.

ml_question → DEFAULT. Use this for general knowledge, out-of-context questions, explanations, theory, casual chat.

IMPORTANT RULES:
- Only choose train_model/prediction/dataset_question if the user is clearly asking to perform an action or query on THEIR uploaded dataset/model.
- If the user asks "explain", "what is", "steps", "how", "why", or any general question, choose ml_question.
- If unsure, choose ml_question.

EXAMPLES:
User: "Who is the president of the United States?" → ml_question
User: "Explain preprocessing steps" → ml_question
User: "Show missing values in my dataset" → dataset_question
User: "How many rows and columns are in my dataset?" → dataset_question
User: "Train a model on my uploaded CSV" → train_model
User: "Predict with age=32 salary=70000 ..." → prediction


USER MESSAGE:
{message}


Return ONLY ONE of these tool names:

train_model
prediction
dataset_question
ml_question
"""

        response = self.llm.ask(prompt)

        tool = response.strip().lower()

        # safer detection
        for t in self.tools:
            if t in tool:
                return t

        return "ml_question"
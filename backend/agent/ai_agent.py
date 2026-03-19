import os

from backend.agent.tool_router import ToolRouter
from backend.services.llm_service import LLMService
from backend.agent.tool_selector import ToolSelector
from backend.memory.chat_memory import ChatMemory
from backend.memory.session_manager import SessionManager


class AIAgent:

    def __init__(self):

        # Core components
        self.tool_router = ToolRouter()
        self.tool_selector = ToolSelector()
        self.llm = LLMService()

        # Memory (per chat thread)
        self.memories = {}
        self.session = SessionManager()

        # Dataset reference
        self.dataset_path = None

    # ---------------------------------
    # Register uploaded dataset
    # ---------------------------------
    def set_dataset(self, dataset_path):

        self.dataset_path = dataset_path

    # ---------------------------------
    # Process user message
    # ---------------------------------
    def get_memory(self, thread_id=None):
        if not thread_id:
            thread_id = "default"
        if thread_id not in self.memories:
            self.memories[thread_id] = ChatMemory()
        return self.memories[thread_id]

    def process_message(self, message, thread_id=None, history=None):

        memory = self.get_memory(thread_id)

        # Store user message in memory
        memory.add_user_message(message)

        msg_lower = (message or "").lower()

        # Select tool using LLM
        intent = self.tool_selector.select_tool(message)
        print("Selected tool:", intent)

        # Guardrail: if no dataset is uploaded, don't route random general questions
        # into dataset/training tools (this was causing "please upload dataset" answers).
        dataset_keywords = ["dataset", "csv", "column", "columns", "row", "rows", "missing", "null", "correlation"]
        train_keywords = ["train", "training", "fit", "build model"]
        predict_keywords = ["predict", "prediction", "infer"]
        looks_dataset_related = any(k in msg_lower for k in dataset_keywords + train_keywords + predict_keywords)

        if not self.dataset_path and intent in {"train_model", "prediction", "dataset_question"} and not looks_dataset_related:
            intent = "ml_question"

        # If user asks to "explain" something (like preprocessing), prefer LLM answers.
        explain_keywords = ["explain", "what is", "steps", "how do", "how to", "why"]
        if intent == "dataset_question" and any(k in msg_lower for k in explain_keywords):
            intent = "ml_question"

        # ---------------------------------
        # TRAIN MODEL
        # ---------------------------------
        if intent == "train_model":

            if not self.dataset_path:

                # Be ChatGPT-like: if user isn't actually requesting training on a dataset,
                # don't block them with an upload message.
                response = (
                    "I can train a model, but I don't have a dataset uploaded yet. "
                    "Upload a CSV first, then say '/train'.\n\n"
                    "Meanwhile, tell me what you want to build (e.g. crop yield predictor, disease predictor) "
                    "and I can guide you step-by-step."
                )

                memory.add_assistant_message(response)

                return {"response": response}

            # Create new session
            self.session.create_session()

            result = self.tool_router.run_tool(
                intent,
                dataset_path=self.dataset_path
            )

            response = str(result)

            memory.add_assistant_message(response)

            return {"response": response}

        # ---------------------------------
        # PREDICTION
        # ---------------------------------
        if intent == "prediction":

            model_path = "models/ml_pipeline.pkl"

            if not os.path.exists(model_path):

                response = (
                    "I can't run prediction yet because no trained model is available. "
                    "Train a model first (upload dataset → /train), then provide features to predict."
                )

                memory.add_assistant_message(response)

                return {"response": response}

            result = self.tool_router.run_tool(
                intent,
                dataset_path=self.dataset_path,
                message=message
            )

            response = str(result)

            memory.add_assistant_message(response)

            return {"response": response}

        # ---------------------------------
        # DATASET QUESTIONS
        # ---------------------------------
        if intent == "dataset_question":

            if not self.dataset_path:

                # If they asked a general question, answer it instead of blocking.
                response = self.llm.ask(
                    message,
                    history=history or memory.get_messages()
                )

                memory.add_assistant_message(response)

                return {"response": response}

            result = self.tool_router.run_tool(
                intent,
                dataset_path=self.dataset_path,
                message=message
            )

            response = str(result)

            memory.add_assistant_message(response)

            return {"response": response}

        # ---------------------------------
        # LLM FALLBACK
        # ---------------------------------

        answer = self.llm.ask(
            message,
            history=history or memory.get_messages()
        )

        memory.add_assistant_message(answer)

        return {"response": answer}


# Global agent instance
agent = AIAgent()
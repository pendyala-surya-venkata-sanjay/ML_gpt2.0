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
        """Get or create memory for a thread."""
        if not thread_id:
            thread_id = "default"
        if thread_id not in self.memories:
            self.memories[thread_id] = ChatMemory(thread_id=thread_id, max_messages=20)
        return self.memories[thread_id]

    def process_message(self, message, thread_id=None, history=None):
        """Process user message with enhanced context awareness."""
        memory = self.get_memory(thread_id)
        
        # Store user message in memory
        memory.add_user_message(message)
        
        msg_lower = (message or "").lower()
        
        # Get conversation context for better routing
        context_summary = memory.get_context_summary()
        
        # Select tool using LLM with enhanced context
        intent = self.tool_selector.select_tool(message, context=context_summary)
        print("Selected tool:", intent, "Context:", context_summary)
        
        # Enhanced guardrail logic with context awareness
        dataset_keywords = ["dataset", "csv", "column", "columns", "row", "rows", "missing", "null", "correlation"]
        train_keywords = ["train", "training", "fit", "build model"]
        predict_keywords = ["predict", "prediction", "infer"]
        explain_keywords = ["explain", "what is", "steps", "how do", "how to", "why"]
        
        looks_dataset_related = any(k in msg_lower for k in dataset_keywords + train_keywords + predict_keywords)
        is_explain_request = any(k in msg_lower for k in explain_keywords)
        
        # Smart routing based on context and intent
        if not self.dataset_path and intent in {"train_model", "prediction", "dataset_question"} and not looks_dataset_related:
            intent = "ml_question"
        
        if intent == "dataset_question" and is_explain_request:
            intent = "ml_question"
        
        # If we have ongoing ML context, prefer dataset-related tools
        if "dataset" in context_summary.lower() and looks_dataset_related and intent == "ml_question":
            intent = "dataset_question"
        
        # ---------------------------------
        # TRAIN MODEL
        # ---------------------------------
        if intent == "train_model":
            if not self.dataset_path:
                response = (
                    "I can help you train a model, but I'll need a dataset first. "
                    "Upload a CSV file, then I can guide you through the training process.\n\n"
                    "In the meantime, what kind of model are you looking to build? "
                    "I can provide guidance on the approach and requirements."
                )
                memory.add_assistant_message(response)
                return {"response": response}
            
            # Create new session for training
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
                    "I can run predictions once we have a trained model. "
                    "Let's first train a model with your dataset, then I can help you make predictions."
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
                # Fall back to LLM for general questions
                response = self.llm.ask(
                    message,
                    history=history or memory.get_recent_messages(10, include_system=False)
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
        # LLM FALLBACK - Enhanced with context
        # ---------------------------------
        
        # Prepare enhanced context for LLM
        enhanced_history = memory.get_recent_messages(10, include_system=False)
        if self.dataset_path:
            # Add context about available dataset
            enhanced_history.append({
                "role": "system",
                "content": f"User has uploaded a dataset at: {self.dataset_path}. ML tools are available."
            })
        
        answer = self.llm.ask(
            message,
            history=enhanced_history
        )
        
        memory.add_assistant_message(answer)
        return {"response": answer}


# Global agent instance
agent = AIAgent()
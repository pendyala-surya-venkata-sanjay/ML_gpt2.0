from backend.chatbot.intent_parser import IntentParser
from backend.chatbot.response_generator import ResponseGenerator
from backend.services.knowledge_service import KnowledgeService


class ConversationManager:

    def __init__(self):

        self.intent_parser = IntentParser()
        self.response_generator = ResponseGenerator()
        self.knowledge = KnowledgeService()

        # store question waiting for user teaching
        self.pending_question = None


    def handle_message(self, message):

        message = message.lower()

        # if bot asked user to teach something
        if self.pending_question is not None:

            self.knowledge.add_new_concept(
                self.pending_question,
                message
            )

            learned_question = self.pending_question
            self.pending_question = None

            return f"Thanks! I learned something new about '{learned_question}'."

        # detect intent
        intent = self.intent_parser.parse(message)

        # explain concept
        if intent == "explain_concept":

            answer = self.knowledge.get_concept(message)

            if answer is None:

                self.pending_question = message

                return "I don't know the answer to that. Can you teach me?"

            return answer


        # training
        elif intent == "train_model":

            return "Training will be triggered through the ML pipeline."


        # prediction
        elif intent == "predict":

            return "Prediction can be performed using the trained model."


        # visualization
        elif intent == "show_visualization":

            return "Visualizations are available in the visualization dashboard."


        # model comparison
        elif intent == "compare_models":

            return "Model comparison results are available in the training dashboard."


        # recommendation
        elif intent == "recommend_model":

            return "The system will recommend the best model based on dataset characteristics."


        # fallback search
        else:

            answer = self.knowledge.get_concept(message)

            if answer is None:

                self.pending_question = message

                return "I couldn't find information about that. Can you teach me?"

            return answer
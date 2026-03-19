import os
from typing import Optional


class LLMService:

    def __init__(self):
        # Important: don't fail backend startup if the key isn't set.
        # We degrade gracefully and return a helpful message from ask().
        self.api_key = os.getenv("GROQ_API_KEY") or None
        self._client = None

    def ask(self, question, history=None):

        try:
            if not self.api_key:
                return (
                    "LLM is not configured. Set the GROQ_API_KEY environment variable on the backend "
                    "to enable ChatGPT-like answers."
                )

            if self._client is None:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional, helpful assistant. "
                        "Write clear, concise answers with a calm, confident tone. "
                        "You can answer general knowledge questions and also provide expert guidance in machine learning. "
                        "When explaining processes, use structured steps and practical guidance. "
                        "If information may be outdated or uncertain, say so briefly and suggest how to verify."
                        "\n\nIf the conversation includes a system message starting with 'CONTEXT', treat it as the user's "
                        "actual dataset/model/training context and answer specifically about that context (not generic theory)."
                    )
                }
            ]

            # Include chat history if available
            if history:
                messages.extend(history)

            messages.append({
                "role": "user",
                "content": question
            })

            completion = self._client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.3
            )

            return completion.choices[0].message.content

        except Exception as e:
            return f"LLM error: {str(e)}"
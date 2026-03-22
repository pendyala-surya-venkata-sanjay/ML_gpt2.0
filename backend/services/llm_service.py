import os
from typing import Optional


class LLMService:

    def __init__(self):
        # Important: don't fail backend startup if the key isn't set.
        # We degrade gracefully and return a helpful message from ask().
        self.api_key = os.getenv("GROQ_API_KEY") or None
        self._client = None

    def ask(self, question, history=None):
        """Enhanced LLM service with better context handling and professional responses."""
        try:
            if not self.api_key:
                return (
                    "I'm currently operating in basic mode. For enhanced AI assistance, "
                    "please configure the GROQ_API_KEY environment variable on the backend."
                )

            if self._client is None:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)

            # Build enhanced system prompt
            system_prompt = (
                "You are a professional AI assistant specializing in machine learning and data science. "
                "You provide clear, structured, and helpful responses with a confident yet approachable tone.\n\n"
                
                "Guidelines:\n"
                "- Be concise but thorough\n"
                "- Use structured formatting (bullet points, numbered lists) when appropriate\n"
                "- Provide practical, actionable advice\n"
                "- Acknowledge limitations and suggest alternatives\n"
                "- Maintain context from previous messages\n"
                "- If discussing ML concepts, be precise but accessible\n\n"
                
                "If the conversation includes system messages about datasets or models, "
                "use that information to provide specific, contextual assistance."
            )

            messages = [{"role": "system", "content": system_prompt}]

            # Include conversation history with context awareness
            if history:
                # Limit history to prevent token overflow but maintain context
                recent_history = history[-10:] if len(history) > 10 else history

                # Normalize history into Groq-compatible format:
                # keep only role/content fields and drop extra keys like timestamp.
                for item in recent_history:
                    if not isinstance(item, dict):
                        continue
                    role = item.get("role")
                    content = item.get("content")
                    if not role or not content:
                        continue
                    messages.append({"role": str(role), "content": str(content)})

            messages.append({
                "role": "user",
                "content": question
            })

            completion = self._client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.2,  # Lower temperature for more consistent responses
                max_tokens=1000   # Reasonable length limit
            )

            response = completion.choices[0].message.content
            
            # Post-process response for better formatting
            if not response.endswith(('.', '!', '?', ':')):
                response += '.'
                
            return response

        except Exception as e:
            # Graceful error handling
            error_msg = f"I encountered an issue processing your request. {str(e)}"
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                error_msg = "I'm currently experiencing high demand. Please try again in a moment."
            return error_msg
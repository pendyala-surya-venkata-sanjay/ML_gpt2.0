import json
import os
from datetime import datetime
from typing import List, Dict, Optional


class ChatMemory:
    """Enhanced chat memory with persistent storage and context management."""

    def __init__(self, thread_id: str = None, max_messages: int = 20):
        self.thread_id = thread_id or "default"
        self.max_messages = max_messages
        self.messages: List[Dict[str, str]] = []
        self.memory_dir = "sessions"
        self._ensure_memory_dir()
        self._load_memory()

    def _ensure_memory_dir(self):
        """Ensure memory directory exists."""
        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir)

    def _get_memory_file(self) -> str:
        """Get the memory file path for this thread."""
        return os.path.join(self.memory_dir, f"{self.thread_id}.json")

    def _load_memory(self):
        """Load existing memory from file."""
        memory_file = self._get_memory_file()
        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.messages = data.get('messages', [])
                    # Trim to max_messages if needed
                    if len(self.messages) > self.max_messages:
                        self.messages = self.messages[-self.max_messages:]
            except (json.JSONDecodeError, IOError):
                # If file is corrupted, start fresh
                self.messages = []

    def _save_memory(self):
        """Save memory to file."""
        memory_file = self._get_memory_file()
        try:
            data = {
                'thread_id': self.thread_id,
                'updated_at': datetime.now().isoformat(),
                'messages': self.messages
            }
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError:
            # If we can't save, continue with in-memory storage
            pass

    def add_user_message(self, message: str):
        """Add a user message to memory."""
        self.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_messages()
        self._save_memory()

    def add_assistant_message(self, message: str):
        """Add an assistant message to memory."""
        self.messages.append({
            "role": "assistant", 
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_messages()
        self._save_memory()

    def add_system_message(self, message: str):
        """Add a system message to memory."""
        self.messages.append({
            "role": "system",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_messages()
        self._save_memory()

    def _trim_messages(self):
        """Keep only the most recent messages within the limit."""
        if len(self.messages) > self.max_messages:
            # Keep the last max_messages messages
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self, include_system: bool = True) -> List[Dict[str, str]]:
        """Get all messages, optionally excluding system messages."""
        if include_system:
            return self.messages.copy()
        else:
            return [msg for msg in self.messages if msg['role'] != 'system']

    def get_recent_messages(self, count: int = 10, include_system: bool = False) -> List[Dict[str, str]]:
        """Get the most recent messages."""
        messages = self.get_messages(include_system=include_system)
        return messages[-count:] if len(messages) > count else messages

    def clear(self):
        """Clear all messages."""
        self.messages = []
        self._save_memory()

    def get_context_summary(self) -> str:
        """Get a summary of conversation context for LLM."""
        if not self.messages:
            return "No previous conversation context."
        
        recent_messages = self.get_recent_messages(6, include_system=False)
        if len(recent_messages) <= 2:
            return "Brief conversation started."
        
        # Create a concise summary
        user_msgs = [msg['content'] for msg in recent_messages if msg['role'] == 'user']
        last_user_msg = user_msgs[-1] if user_msgs else ""
        
        if "dataset" in last_user_msg.lower() or "train" in last_user_msg.lower():
            return "User is working on ML dataset training."
        elif "predict" in last_user_msg.lower():
            return "User is asking for predictions."
        elif len(user_msgs) >= 2:
            return f"Ongoing conversation ({len(user_msgs)} user messages). Last topic: {last_user_msg[:100]}..."
        else:
            return "Single question conversation."

    def set_thread_id(self, thread_id: str):
        """Switch to a different thread."""
        if self.thread_id != thread_id:
            self._save_memory()  # Save current thread
            self.thread_id = thread_id
            self.messages = []
            self._load_memory()  # Load new thread
import os
import uuid


class SessionManager:

    def __init__(self):

        self.base_dir = "sessions"

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.current_session = None

    def create_session(self):

        session_id = "session_" + str(uuid.uuid4())[:8]

        path = os.path.join(self.base_dir, session_id)

        os.makedirs(path)

        self.current_session = path

        return path

    def get_current_session(self):

        return self.current_session
## ML Assistant (Frontend + FastAPI)

### What this is
- **Backend**: FastAPI API for dataset upload, training, prediction, and chat.
- **Frontend**: ChatGPT-like web UI (React + Tailwind) that talks to the backend.

### Run the backend
1. Create/activate a virtualenv.
2. Install Python deps:

```bash
pip install -r requirements.txt
```

3. Set your Groq API key (required for `/chat` tool selection + ML Q&A):

```bash
setx GROQ_API_KEY "YOUR_KEY"
```

4. Start FastAPI:

```bash
uvicorn backend.main:app --reload
```

Backend default: `http://127.0.0.1:8000`

### Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend default: `http://127.0.0.1:5173`

### Configure API base URL (optional)
Create `frontend/.env`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from backend.api.chat_routes import router as chat_router
from backend.api.dataset_routes import router as dataset_router
from backend.api.analysis_routes import router as analysis_router
from backend.api.models_routes import router as models_router
from backend.api.training_routes import router as training_router
from backend.api.prediction_routes import router as prediction_router


app = FastAPI(
    title="AI ML Assistant",
    description="Machine Learning Assistant API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated visualization images over HTTP.
# The ML pipeline writes PNGs into ./visualizations
os.makedirs("visualizations", exist_ok=True)
app.mount("/static/visualizations", StaticFiles(directory="visualizations"), name="visualizations")
os.makedirs("generated_projects", exist_ok=True)
app.mount("/static/exports", StaticFiles(directory="generated_projects"), name="exports")

app.include_router(chat_router)
app.include_router(dataset_router)
app.include_router(analysis_router)
app.include_router(models_router)
app.include_router(training_router)
app.include_router(prediction_router)


@app.get("/")
def home():
    return {"message": "ML Assistant API is running"}
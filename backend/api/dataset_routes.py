from fastapi import APIRouter, UploadFile, File
import shutil
import os

from backend.services.logger import logger
from backend.agent.ai_agent import agent   # IMPORTANT: import the global agent

router = APIRouter()

UPLOAD_FOLDER = "datasets/uploaded_datasets"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):

    try:

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Register dataset inside AI agent
        agent.set_dataset(file_path)

        logger.info(f"Dataset uploaded successfully: {file.filename}")

        return {
            "status": "success",
            "message": "Dataset uploaded successfully",
            "dataset_path": file_path
        }

    except Exception as e:

        logger.error(f"Dataset upload failed: {str(e)}")

        return {
            "status": "error",
            "message": str(e)
        }
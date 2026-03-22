from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
from pathlib import Path

from backend.services.logger import logger
from backend.agent.ai_agent import agent   # IMPORTANT: import the global agent

router = APIRouter()

# Resolve paths relative to the repo root so uploads work regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_FOLDER = _REPO_ROOT / "datasets" / "uploaded_datasets"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):

    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing uploaded filename")

        # Avoid path traversal and normalize to a plain filename.
        safe_name = os.path.basename(file.filename).replace("\\", "_").replace("/", "_").strip()
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid uploaded filename")

        # Keep behavior aligned with the UI (CSV upload).
        if not safe_name.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only .csv datasets are supported")

        file_path = Path(UPLOAD_FOLDER) / safe_name

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Register dataset inside AI agent
        agent.set_dataset(str(file_path))

        logger.info(f"Dataset uploaded successfully: {file.filename}")

        # Return a normalized path that won't create invalid escape sequences later.
        dataset_path = str(file_path).replace("\\", "/")

        return {
            "status": "success",
            "message": "Dataset uploaded successfully",
            "dataset_path": dataset_path
        }

    except HTTPException as e:
        logger.error(f"Dataset upload failed: {e.detail}")
        return {"status": "error", "message": e.detail}

    except Exception as e:
        logger.error(f"Dataset upload failed: {str(e)}")

        return {
            "status": "error",
            "message": str(e)
        }
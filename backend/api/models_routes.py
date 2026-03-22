from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.logger import logger
from backend.services import model_registry
from ml_pipeline.export_project import ProjectExporter

router = APIRouter()


class DeleteByDatasetRequest(BaseModel):
    dataset_path: str


class DeleteModelRequest(BaseModel):
    model_id: str


@router.get("/models")
def get_models():
    return {
        "status": "success",
        "active_model_id": model_registry.get_active_model_id(),
        "models": model_registry.list_models(),
    }


@router.post("/models/active")
def set_active_model(model_id: str):
    try:
        ok = model_registry.set_active_model(model_id)
        if not ok:
            return {"status": "error", "message": "Model not found"}
        return {"status": "success", "active_model_id": model_id}
    except Exception as e:
        logger.error(f"Set active model failed: {str(e)}")
        return {"status": "error", "message": str(e)}


@router.get("/exports")
def get_exports():
    return {"status": "success", "exports": model_registry.list_exports()}


@router.post("/models/rename")
def rename_model(model_id: str, display_name: str):
    try:
        ok = model_registry.rename_model(model_id, display_name)
        if not ok:
            return {"status": "error", "message": "Model not found"}
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Rename model failed: {str(e)}")
        return {"status": "error", "message": str(e)}


@router.post("/exports/create")
def create_export(model_id: str, project_name: str):
    """
    Re-export a selected model with a user-defined project/zip name.
    """
    try:
        m = model_registry.get_model_by_id(model_id)
        if not m or not m.get("pipeline_path"):
            return {"status": "error", "message": "Model not found"}

        exporter = ProjectExporter()
        zip_path = exporter.export_project(model_path=m["pipeline_path"], project_name=project_name)
        model_registry.add_export(model_id, zip_path)
        return {"status": "success", "zip_path": zip_path}
    except Exception as e:
        logger.error(f"Export create failed: {str(e)}")
        return {"status": "error", "message": str(e)}


@router.post("/models/clear")
def clear_models():
    """
    Clear all registered models/exports and generated artifacts.
    Intended for development resets.
    """
    try:
        model_registry.clear_registry_and_files()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Clear models failed: {str(e)}")
        return {"status": "error", "message": str(e)}


@router.post("/models/delete_by_dataset")
def delete_models_by_dataset(payload: DeleteByDatasetRequest):
    """
    Remove all trained models/exports associated with the given dataset path.
    Used when the user deletes a chat.
    """

    try:
        result = model_registry.delete_models_by_dataset_path(payload.dataset_path)
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"Delete-by-dataset failed: {str(e)}")
        return {"status": "error", "message": str(e)}


@router.post("/models/delete")
def delete_model(payload: DeleteModelRequest):
    """
    Delete a single trained model and all its artifacts:
    pipeline file, exported zip(s), dataset (best-effort), and visualizations.
    """

    try:
        result = model_registry.delete_model_by_id(payload.model_id)
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"Delete model failed: {str(e)}")
        return {"status": "error", "message": str(e)}




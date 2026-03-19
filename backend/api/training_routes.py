from fastapi import APIRouter
from ml_pipeline.ml_pipeline_controller import MLPipelineController
from backend.services.logger import logger
from backend.services import model_registry
from backend.services.json_sanitize import sanitize_for_json
from backend.agent.ai_agent import agent
import os
import uuid

router = APIRouter()


@router.post("/train_model")
def train_model(dataset_path: str, model_name: str = None, export_name: str = None):

    try:

        logger.info(f"Training started for dataset: {dataset_path}")

        # Ensure the conversational agent remembers the current dataset,
        # even if training was triggered outside /upload_dataset.
        agent.set_dataset(dataset_path)

        pipeline = MLPipelineController(dataset_path)

        result = pipeline.run_pipeline()

        logger.info("Training completed")

        # Register the trained model so user can select it later.
        model_id = "model_" + str(uuid.uuid4())[:8]
        dataset_name = os.path.basename(dataset_path)
        analysis = result.get("analysis") or {}
        training = result.get("training") or {}
        evaluation = result.get("evaluation")

        pipeline_path = None
        try:
            pipeline_path = training.get("pipeline_path")
        except Exception:
            pipeline_path = None

        if pipeline_path:
            model_registry.register_model(
                model_id=model_id,
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                problem_type=analysis.get("problem_type"),
                target_column=analysis.get("target_column"),
                best_model_name=training.get("best_model"),
                metrics=evaluation,
                pipeline_path=pipeline_path,
                export_zip_path=result.get("export_path"),
                display_name=model_name,
            )

        payload = {
            "status": "success",
            "model_id": model_id,
            "logs": result["logs"],
            "analysis": result.get("analysis"),
            "visualizations": result.get("visualizations"),
            "training_results": result["training"],
            "evaluation": result["evaluation"],
            "pipeline_code": result.get("pipeline_code"),
            "export_path": result["export_path"]
        }
        return sanitize_for_json(payload)

    except Exception as e:
        import traceback
        error_details = f"Training failed: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
        logger.error(error_details)

        return sanitize_for_json({
            "status": "error",
            "message": str(e),
            "full_error": error_details
        })
from fastapi import APIRouter

from backend.services.logger import logger
from backend.services.json_sanitize import sanitize_for_json
from ml_pipeline.analyze_data import DatasetAnalyzer

router = APIRouter()


@router.get("/analyze_dataset")
def analyze_dataset(dataset_path: str):
    """
    Returns a rich dataset analysis report (target detection, problem type, missing values, correlations, preview).
    Designed for chat-driven, step-by-step ML workflows.
    """
    try:
        analyzer = DatasetAnalyzer(dataset_path)
        result = analyzer.analyze()
        return sanitize_for_json({"status": "success", "analysis": result})
    except Exception as e:
        logger.error(f"Dataset analysis failed: {str(e)}")
        return sanitize_for_json({"status": "error", "message": str(e)})


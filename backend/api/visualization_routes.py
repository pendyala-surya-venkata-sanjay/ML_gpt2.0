from fastapi import APIRouter
import os
import zipfile
from datetime import datetime

from backend.services.logger import logger


router = APIRouter()


@router.get("/visualizations/export_all")
def export_all_visualizations():
  """
  Package all generated visualization images into a single zip and return its backend path.
  The zip is written into generated_projects so it can be downloaded via /static/exports.
  """
  try:
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)

    image_files = [
      f for f in os.listdir(viz_dir)
      if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    if not image_files:
      return {"status": "error", "message": "No visualizations available yet"}

    exports_dir = "generated_projects"
    os.makedirs(exports_dir, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    zip_name = f"visualizations_{ts}.zip"
    zip_path = os.path.join(exports_dir, zip_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
      for filename in image_files:
        full_path = os.path.join(viz_dir, filename)
        if os.path.isfile(full_path):
          zf.write(full_path, arcname=filename)

    logger.info(f"Visualization archive created: {zip_path}")
    return {"status": "success", "zip_path": zip_path}
  except Exception as e:
    logger.error(f"Visualization export failed: {str(e)}")
    return {"status": "error", "message": str(e)}


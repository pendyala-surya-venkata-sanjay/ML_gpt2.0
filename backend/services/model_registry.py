import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional


REGISTRY_PATH = os.path.join("models", "registry.json")


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_dir():
    os.makedirs("models", exist_ok=True)


def load_registry() -> Dict[str, Any]:
    _ensure_dir()
    if not os.path.exists(REGISTRY_PATH):
        return {"active_model_id": None, "models": [], "exports": []}
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("active_model_id", None)
        data.setdefault("models", [])
        data.setdefault("exports", [])
        return data
    except Exception:
        return {"active_model_id": None, "models": [], "exports": []}


def save_registry(data: Dict[str, Any]) -> None:
    _ensure_dir()
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def list_models() -> List[Dict[str, Any]]:
    return load_registry().get("models", [])


def get_active_model_id() -> Optional[str]:
    return load_registry().get("active_model_id")


def set_active_model(model_id: str) -> bool:
    reg = load_registry()
    if not any(m.get("id") == model_id for m in reg.get("models", [])):
        return False
    reg["active_model_id"] = model_id
    save_registry(reg)
    return True


def register_model(
    *,
    model_id: str,
    dataset_path: str,
    dataset_name: str,
    problem_type: str,
    target_column: str,
    best_model_name: str,
    metrics: Any,
    pipeline_path: str,
    export_zip_path: Optional[str] = None,
    display_name: Optional[str] = None,
) -> Dict[str, Any]:
    reg = load_registry()

    if display_name is None:
        display_name = os.path.splitext(dataset_name)[0]

    entry = {
        "id": model_id,
        "created_at": _now_iso(),
        "display_name": display_name,
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "problem_type": problem_type,
        "target_column": target_column,
        "best_model": best_model_name,
        "metrics": metrics,
        "pipeline_path": pipeline_path,
    }

    reg["models"].insert(0, entry)
    reg["active_model_id"] = model_id

    if export_zip_path:
        reg.setdefault("exports", [])
        reg["exports"].insert(
            0,
            {
                "model_id": model_id,
                "created_at": _now_iso(),
                "zip_path": export_zip_path,
            },
        )

    save_registry(reg)
    return entry


def list_exports() -> List[Dict[str, Any]]:
    return load_registry().get("exports", [])


def get_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    for m in load_registry().get("models", []):
        if m.get("id") == model_id:
            return m
    return None


def _normalize_path_for_match(path: Optional[str]) -> str:
    """
    Normalize dataset/pipeline paths for matching on Windows.
    This prevents misses due to `\\` vs `/`, casing, and trailing slashes.
    """
    if not path:
        return ""
    s = str(path).strip().replace("\\", "/")
    # Windows paths are case-insensitive for drive letter and typically for folders.
    s = s.lower()
    return s.rstrip("/")


def delete_models_by_dataset_path(dataset_path: str) -> Dict[str, Any]:
    """
    Delete all registered models/exports whose dataset_path matches.
    This is used to remove an entire chat's trained artifacts.
    """

    reg = load_registry()
    models = reg.get("models", [])
    exports = reg.get("exports", [])
    active_model_id = reg.get("active_model_id")

    normalized_target = _normalize_path_for_match(dataset_path)
    matching_models = [
        m for m in models if _normalize_path_for_match(m.get("dataset_path")) == normalized_target
    ]
    removed_model_ids = {m.get("id") for m in matching_models if m.get("id")}

    def _safe_remove(path: Optional[str]) -> None:
        if not path:
            return
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            # Best-effort delete: never fail the whole endpoint for missing files.
            pass

    removed_pipeline_files = 0
    for m in matching_models:
        _safe_remove(m.get("pipeline_path"))
        removed_pipeline_files += 1

    # Remove related exports.
    remaining_exports: List[Dict[str, Any]] = []
    removed_export_files = 0
    for ex in exports:
        if ex.get("model_id") in removed_model_ids:
            _safe_remove(ex.get("zip_path"))
            removed_export_files += 1
        else:
            remaining_exports.append(ex)

    reg["models"] = [m for m in models if m.get("id") not in removed_model_ids]
    reg["exports"] = remaining_exports

    if active_model_id in removed_model_ids:
        reg["active_model_id"] = None

    # Optionally delete the dataset itself when no remaining model references it.
    deleted_dataset = False
    try:
        remaining_models = reg.get("models", [])
        remaining_refs = [
            m for m in remaining_models if _normalize_path_for_match(m.get("dataset_path")) == normalized_target
        ]
        if normalized_target and not remaining_refs:
            if os.path.exists(dataset_path):
                os.remove(dataset_path)
                deleted_dataset = True
    except Exception:
        deleted_dataset = False

    save_registry(reg)

    return {
        "removed_models": len(matching_models),
        "removed_exports": removed_export_files,
        "removed_pipeline_files": removed_pipeline_files,
        "deleted_dataset": deleted_dataset,
    }


def delete_model_by_id(model_id: str) -> Dict[str, Any]:
    """
    Delete a single trained model and all its registered artifacts.
    Best-effort cleanup of related generated files and datasets.
    """

    reg = load_registry()
    models = reg.get("models", [])
    exports = reg.get("exports", [])

    target = next((m for m in models if m.get("id") == model_id), None)
    if not target:
        return {"removed_models": 0}

    dataset_path = target.get("dataset_path")
    pipeline_path = target.get("pipeline_path")

    target_export_entries = [ex for ex in exports if ex.get("model_id") == model_id]
    removed_export_files = 0

    def _safe_remove(path: Optional[str]) -> None:
        if not path:
            return
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    # Remove pipeline file.
    _safe_remove(pipeline_path)

    # Remove export zip(s).
    for ex in target_export_entries:
        if ex.get("zip_path"):
            _safe_remove(ex.get("zip_path"))
            removed_export_files += 1

    removed_models_count = 1

    # Update registry first, then decide whether dataset should be removed.
    remaining_models = [m for m in models if m.get("id") != model_id]
    remaining_exports = [ex for ex in exports if ex.get("model_id") != model_id]
    reg["models"] = remaining_models
    reg["exports"] = remaining_exports

    active_model_id = reg.get("active_model_id")
    if active_model_id == model_id:
        reg["active_model_id"] = None

    save_registry(reg)

    # Delete dataset only if no other models still reference it.
    normalized_target = _normalize_path_for_match(dataset_path)
    remaining_dataset_refs = [
        m for m in remaining_models if _normalize_path_for_match(m.get("dataset_path")) == normalized_target
    ]
    deleted_dataset = False
    if dataset_path and not remaining_dataset_refs:
        try:
            if os.path.exists(dataset_path):
                os.remove(dataset_path)
                deleted_dataset = True
        except Exception:
            deleted_dataset = False

    # Visualizations are generated into a shared folder (overwritten on train),
    # so we clear them for cleanup when deleting a model.
    deleted_visualizations = False
    try:
        shutil.rmtree("visualizations", ignore_errors=True)
        os.makedirs("visualizations", exist_ok=True)
        deleted_visualizations = True
    except Exception:
        deleted_visualizations = False

    return {
        "removed_models": removed_models_count,
        "removed_exports": removed_export_files,
        "deleted_dataset": deleted_dataset,
        "deleted_visualizations": deleted_visualizations,
    }


def rename_model(model_id: str, display_name: str) -> bool:
    reg = load_registry()
    changed = False
    for m in reg.get("models", []):
        if m.get("id") == model_id:
            m["display_name"] = display_name
            changed = True
            break
    if changed:
        save_registry(reg)
    return changed


def add_export(model_id: str, zip_path: str) -> Dict[str, Any]:
    reg = load_registry()
    entry = {"model_id": model_id, "created_at": _now_iso(), "zip_path": zip_path}
    reg.setdefault("exports", [])
    reg["exports"].insert(0, entry)
    save_registry(reg)
    return entry


def clear_registry_and_files() -> None:
    """
    Remove all registered models/exports and delete generated artifacts.
    Safe to call when you want a fresh start.
    """
    # Reset registry
    if os.path.exists(REGISTRY_PATH):
        os.remove(REGISTRY_PATH)

    # Helper to clean and recreate a folder
    def _reset_dir(path: str) -> None:
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

    # Remove trained pipelines and registry folder contents
    _reset_dir("models")

    # Remove generated project zips and visualization images
    _reset_dir("generated_projects")
    _reset_dir("visualizations")

    # Optionally clear uploaded datasets (user can re-upload)
    _reset_dir(os.path.join("datasets", "uploaded_datasets"))


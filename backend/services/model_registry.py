import json
import os
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


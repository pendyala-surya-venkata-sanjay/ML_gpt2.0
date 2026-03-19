import math
from numbers import Number
from typing import Any


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert values that are not JSON-compliant (NaN, +/-Inf) into None.
    Also converts numpy scalar types to native Python where possible.
    """
    # Fast path for common scalars
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    # Any numeric type that can represent NaN/Inf (float, numpy floats, Decimal, etc.)
    if isinstance(obj, Number):
        try:
            f = float(obj)
            if math.isnan(f) or math.isinf(f):
                return None
            # Keep ints as ints where possible
            if isinstance(obj, (int,)) and not isinstance(obj, bool):
                return int(obj)
            return f if isinstance(obj, float) or not float(obj).is_integer() else obj
        except Exception:
            # If it can't be converted, fall through
            pass

    # numpy scalars (optional dependency)
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.generic):
            return sanitize_for_json(obj.item())
        if isinstance(obj, np.ndarray):
            return sanitize_for_json(obj.tolist())
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]

    # Fallback: attempt to stringify unknown objects
    return str(obj)


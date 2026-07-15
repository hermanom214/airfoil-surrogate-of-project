from __future__ import annotations

import json

import numpy as np

from src.evaluate_io import sanitize_for_json


def test_sanitize_for_json_converts_nan_and_inf_to_none() -> None:
    assert sanitize_for_json(float("nan")) is None
    assert sanitize_for_json(float("inf")) is None
    assert sanitize_for_json(float("-inf")) is None


def test_sanitize_for_json_handles_numpy_scalars_and_nested_values() -> None:
    payload = {
        "scalar": np.float32(1.5),
        "nested": {
            "nan": np.float64(np.nan),
            "arr": np.array([1.0, np.inf, 2.0], dtype=np.float64),
            "tuple": (np.int32(3), np.float32(4.5)),
        },
    }

    sanitized = sanitize_for_json(payload)

    assert sanitized["scalar"] == 1.5
    assert sanitized["nested"]["nan"] is None
    assert sanitized["nested"]["arr"] == [1.0, None, 2.0]
    assert sanitized["nested"]["tuple"] == [3, 4.5]

    # Must serialize in strict JSON mode without NaN/Infinity.
    json.dumps(sanitized, allow_nan=False)

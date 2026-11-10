"""Validate user-supplied strand calibration, offline and without app/DB writes.

Create a blank input with ``python -m
pasta_weight_app_v2.experiments.product_calibration --template``. Fill all six
fields from a real product and a counted, weighed sample, then pass that JSON
file as the positional argument. The output is a review draft, not an approved
calibration and not an image-based weight prediction. Keep measured profiles in
the ignored private data directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


TEMPLATE = {
    "product_name": None,
    "diameter_mm": None,
    "length_mm": None,
    "measured_strand_count": None,
    "measured_total_weight_g": None,
    "scale_resolution_g": None,
}


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite positive number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return number


def validate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Check measurements and derive mean mass; never grant approval.

    The interval accounts *only* for rounding one net-weight display to the
    nearest resolution increment. Tare error, instrument accuracy, drift,
    moisture, product variation and counting errors are not included. It is
    neither a confidence interval nor the uncertainty of a future image result.
    """
    if not isinstance(profile, dict) or set(profile) != set(TEMPLATE):
        raise ValueError("Provide exactly the six fields in --template")
    name = profile["product_name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("product_name must identify the measured product")
    count = profile["measured_strand_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("measured_strand_count must be a positive integer")
    measurements = {"product_name": name.strip(), "measured_strand_count": count}
    for field in (
        "diameter_mm", "length_mm", "measured_total_weight_g", "scale_resolution_g"
    ):
        measurements[field] = _positive_number(profile[field], field)
    total = measurements["measured_total_weight_g"]
    resolution = measurements["scale_resolution_g"]
    if total < resolution:
        raise ValueError("The measured total must be at least one scale increment")
    try:
        mean = total / count
        rounding_half_width = resolution / count / 2
    except OverflowError as exc:
        raise ValueError("Strand count is too large for numerical calculation") from exc
    if not math.isfinite(mean) or mean <= 0:
        raise ValueError("Mean strand weight cannot be represented as a positive number")
    interval = [max(0.0, mean - rounding_half_width), mean + rounding_half_width]
    if rounding_half_width <= 0 or not all(math.isfinite(value) for value in interval):
        raise ValueError("Rounding interval cannot be represented with these measurements")
    return {
        "schema_version": 1,
        "status": "INPUT_VALIDATED_REVIEW_REQUIRED",
        "approved": False,
        "production_eligible": False,
        "measurements": measurements,
        "mean_weight_per_strand_g": mean,
        "rounding_only_half_width_g_per_strand": rounding_half_width,
        "rounding_only_interval_g_per_strand": interval,
        "rounding_assumption": "One net-weight display rounded to the nearest scale increment",
        "total_measurement_uncertainty_g": None,
        "image_prediction_uncertainty_g": None,
        "limitations": [
            "Input validation does not verify the product, count or scale measurement.",
            "Rounding interval excludes tare error, scale accuracy, drift and counting error.",
            "Use intact dry strands of the stated product, diameter and length; repeat measurements.",
            "A valid mean strand weight does not validate image-derived strand counts.",
        ],
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def load_profile(path: str | Path) -> dict[str, Any]:
    """Read without modifying the source; record SHA-256 for future provenance."""
    raw = Path(path).read_bytes()
    result = validate_profile(json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object))
    result["source_sha256"] = hashlib.sha256(raw).hexdigest()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?", help="Private calibration input JSON")
    parser.add_argument("--template", action="store_true", help="Print an unfilled template")
    args = parser.parse_args(argv)
    if args.template == bool(args.profile):
        parser.error("Choose either --template or one profile file")
    try:
        result = TEMPLATE if args.template else load_profile(args.profile)
    except (OSError, ValueError, UnicodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

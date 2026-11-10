"""Offline numerical calibration contracts, using synthetic fixture values."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pasta_weight_app_v2.experiments.product_calibration import (
    TEMPLATE, load_profile, main, validate_profile,
)


@pytest.fixture
def measured_profile():
    # Synthetic numbers exercise arithmetic; this is not a real product profile.
    return dict(product_name="SYNTHETIC_TEST_ONLY", diameter_mm=1.6, length_mm=250,
                measured_strand_count=50, measured_total_weight_g=32.0, scale_resolution_g=1.0)


def test_mean_and_rounding_do_not_claim_approval_or_prediction_accuracy(measured_profile):
    before = copy.deepcopy(measured_profile)
    result = validate_profile(measured_profile)
    assert result["mean_weight_per_strand_g"] == pytest.approx(0.64)
    assert result["rounding_only_interval_g_per_strand"] == pytest.approx([0.63, 0.65])
    assert result["rounding_only_half_width_g_per_strand"] == pytest.approx(0.01)
    assert result["total_measurement_uncertainty_g"] is None
    assert result["image_prediction_uncertainty_g"] is None
    assert not result["approved"] and not result["production_eligible"]
    assert measured_profile == before


@pytest.mark.parametrize("field,value", [
    ("product_name", "  "), ("product_name", 123),
    ("diameter_mm", 0), ("diameter_mm", float("nan")),
    ("length_mm", -1), ("length_mm", True),
    ("measured_strand_count", 0), ("measured_strand_count", -3),
    ("measured_strand_count", 50.5), ("measured_strand_count", True),
    ("measured_strand_count", "50"),
    ("measured_total_weight_g", float("inf")), ("measured_total_weight_g", None),
    ("measured_total_weight_g", 0.1),
    ("scale_resolution_g", 0), ("scale_resolution_g", -1),
])
def test_invalid_or_unmeasurable_inputs_are_rejected(measured_profile, field, value):
    measured_profile[field] = value
    with pytest.raises(ValueError):
        validate_profile(measured_profile)


def test_no_fallback_calibration_or_caller_approval(measured_profile):
    with pytest.raises(ValueError):
        validate_profile(TEMPLATE)
    missing = dict(measured_profile)
    del missing["measured_total_weight_g"]
    with pytest.raises(ValueError):
        validate_profile(missing)
    measured_profile["approved"] = True
    with pytest.raises(ValueError):
        validate_profile(measured_profile)


def test_finite_inputs_with_unrepresentable_derived_values_are_rejected(measured_profile):
    measured_profile.update(measured_strand_count=1,
                            measured_total_weight_g=sys.float_info.max,
                            scale_resolution_g=sys.float_info.max)
    with pytest.raises(ValueError, match="interval"):
        validate_profile(measured_profile)


def test_source_file_is_preserved_and_identified(tmp_path, measured_profile):
    path = tmp_path / "synthetic.json"
    raw = json.dumps(measured_profile).encode()
    path.write_bytes(raw)
    result = load_profile(path)
    assert result["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw
    assert list(tmp_path.iterdir()) == [path]


def test_ambiguous_duplicate_json_measurement_is_rejected(tmp_path, measured_profile):
    path = tmp_path / "ambiguous.json"
    raw = json.dumps(measured_profile)[:-1] + ', "measured_strand_count": 100}'
    path.write_text(raw)
    with pytest.raises(ValueError, match="Duplicate"):
        load_profile(path)


def test_cli_template_and_validated_output_are_drafts(tmp_path, measured_profile, capsys):
    assert main(["--template"]) == 0
    assert all(value is None for value in json.loads(capsys.readouterr().out).values())
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(measured_profile))
    assert main([str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "INPUT_VALIDATED_REVIEW_REQUIRED"
    assert not result["production_eligible"]


def test_cli_reports_input_error_instead_of_fabricating_a_profile(tmp_path, capsys):
    path = tmp_path / "blank.json"
    path.write_text(json.dumps(TEMPLATE))
    with pytest.raises(SystemExit) as exc:
        main([str(path)])
    assert exc.value.code == 2
    assert capsys.readouterr().out == ""

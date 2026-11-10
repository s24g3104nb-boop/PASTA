"""Synthetic-only contracts for offline mask research, never accuracy tests."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pasta_weight_app_v2.experiments import compare_masks as comparison
from pasta_weight_app_v2.pasta_vision import detect_bundle_ends, detect_spread


MODES = ("spread_overlap", "bundle_ends_low_angle", "handheld_bundle_ends")


def _annotation(supported=True):
    return {
        "roi_polygon_normalized": [[0, 0], [1, 0], [1, 1], [0, 1]] if supported else [],
        "positive_points": [[0.5, 0.5]],
        "negative_points": [[0.05, 0.05]],
        "observed_view": "synthetic_fixture" if supported else "oblique_side_not_supported",
        "status": "DIAGNOSTIC_DRAFT_NOT_APPROVED",
        "full_mask_ground_truth": False,
    }


def _save_labels(root, rows):
    with (root / "labels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def series(tmp_path):
    root = tmp_path / "originals"
    root.mkdir()
    rows, annotations = [], {}
    for weight in (20, 40, 60, 80, 100):
        for mode in MODES:
            identifier = f"synthetic_{weight}_{mode}"
            path = root / f"{identifier}.png"
            # Blank fixtures test data contracts, not photographic segmentation quality.
            Image.new("RGB", (64, 48), "black").save(path)
            rows.append({
                "image_path": path.name,
                "original_filename": path.name,
                "nominal_weight_g": str(weight),
                "capture_mode": mode,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "physical_family_group_id": "SYNTHETIC_ONE_FAMILY",
            })
            annotations[identifier] = _annotation(not (weight == 60 and mode == MODES[2]))
    _save_labels(root, rows)
    annotation_path = tmp_path / "draft.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    return root, annotation_path, rows, tmp_path / "report"


@pytest.mark.parametrize("mode", MODES)
def test_hsv_baseline_matches_application_and_preserves_input(mode):
    rgb = np.zeros((160, 240, 3), np.uint8)
    if mode == MODES[0]:
        for y in (30, 70, 110):
            cv2.line(rgb, (20, y), (220, y), (235, 190, 65), 5)
    else:
        for xy in ((40, 40), (100, 40), (160, 40), (70, 100), (150, 100)):
            cv2.circle(rgb, xy, 12, (235, 190, 65), -1)
    before = rgb.copy()
    params = comparison.runtime_parameters()
    masks = comparison.masks(rgb, mode, params)
    production = (detect_spread(rgb, **params["DEFAULT_SPREAD"]) if mode == MODES[0]
                  else detect_bundle_ends(rgb, **params["DEFAULT_BUNDLE"]))
    assert production.count > 0
    assert np.array_equal(masks["hsv_baseline"], production.mask)
    features, _, _ = comparison.feature_row(
        masks["hsv_baseline"], rgb, "hsv_baseline", mode, _annotation(), params, 1.0
    )
    actual_count = features["legacy_detected_count"] if mode == MODES[0] else features["circle_candidates"]
    assert actual_count == production.count
    assert np.array_equal(rgb, before)
    assert features["predicted_weight_g"] is None
    assert features["training_eligible"] is False
    assert features["mask_qc"] == "NOT_VERIFIED"


def test_unknown_capture_state_is_not_silently_mapped_to_bundle():
    with pytest.raises(ValueError):
        comparison.masks(np.zeros((64, 64, 3), np.uint8), "cooked_pasta", comparison.runtime_parameters())


@pytest.mark.parametrize("points", [
    [[0, 0], [1, 1]],
    [[-0.1, 0], [1, 0], [1, 1]],
    [[0, 0], [float("nan"), 0], [1, 1]],
    [[0, 0], [1, 0], [1, 1.2]],
])
def test_invalid_manual_roi_is_rejected(points):
    with pytest.raises(ValueError):
        comparison.roi_mask((64, 64, 3), {"roi_polygon_normalized": points})


def test_watershed_labels_cannot_turn_background_into_pasta():
    mask = np.zeros((512, 512), np.uint8)
    cv2.circle(mask, (230, 256), 45, 255, -1)
    cv2.circle(mask, (285, 256), 45, 255, -1)
    rgb = np.zeros((512, 512, 3), np.uint8)
    rgb[mask > 0] = (235, 190, 65)
    original_mask = mask.copy()
    labels, regions = comparison.split_regions(mask, rgb)
    assert regions >= 1
    assert np.all(labels[mask == 0] == 0)
    assert np.array_equal(mask, original_mask)
    # A segmentation label is not a strand count or a mass estimate.
    features, _, _ = comparison.feature_row(
        mask, rgb, "watershed", MODES[0], _annotation(), comparison.runtime_parameters(), 1.0
    )
    assert features["legacy_detected_count"] is None
    assert features["predicted_weight_g"] is None


@pytest.mark.parametrize("problem", [
    "second_family", "empty_family", "wrong_hash", "outside_path",
    "duplicate_id", "incomplete_matrix", "invalid_annotation",
])
def test_invalid_receipt_rejected_before_output_creation(series, problem):
    root, draft, rows, output = series
    if problem == "second_family":
        rows[-1]["physical_family_group_id"] = "SECOND_FAMILY"
    elif problem == "empty_family":
        for row in rows:
            row["physical_family_group_id"] = ""
    elif problem == "wrong_hash":
        rows[-1]["sha256"] = "0" * 64
    elif problem == "outside_path":
        outside = root.parent / "outside.png"
        outside.write_bytes((root / rows[-1]["image_path"]).read_bytes())
        rows[-1]["image_path"] = "../outside.png"
    elif problem == "duplicate_id":
        rows[-1]["original_filename"] = rows[0]["original_filename"]
    elif problem == "incomplete_matrix":
        rows[-1]["nominal_weight_g"] = "20"
    else:
        annotations = json.loads(draft.read_text(encoding="utf-8"))
        identifier = Path(rows[-1]["original_filename"]).stem
        annotations[identifier]["roi_polygon_normalized"][0] = [-2, 0]
        draft.write_text(json.dumps(annotations), encoding="utf-8")
    _save_labels(root, rows)
    with pytest.raises(ValueError):
        comparison.run(root, draft, output)
    assert not output.exists()


def test_existing_output_is_never_overwritten(series):
    root, draft, _, output = series
    output.mkdir()
    marker = output / "prior-report.txt"
    marker.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        comparison.run(root, draft, output)
    assert marker.read_text(encoding="utf-8") == "keep me"


def test_full_report_keeps_single_family_unapproved_and_unsupported_distinct(series, monkeypatch):
    import sqlite3

    root, draft, _, output = series
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}

    def forbid_database_access(*args, **kwargs):
        pytest.fail("Offline mask comparison must not open an application database")

    monkeypatch.setattr(sqlite3, "connect", forbid_database_access)
    report = comparison.run(root, draft, output)
    assert report["images"] == 15 and report["families"] == 1
    assert report["training_images"] == report["independent_test_images"] == 0
    assert report["calibration_applied"] is False and report["accuracy_metrics"] is None
    assert report["annotation_status"] == "DRAFT_NOT_APPROVED"
    assert len(report["rows"]) == 15 * 8 * 3
    assert {r["scale"] for r in report["rows"]} == {1.0, 0.75, 0.5}
    assert all(r["predicted_weight_g"] is None and not r["training_eligible"] for r in report["rows"])
    unsupported = [r for r in report["rows"] if r["status"] == "UNSUPPORTED_SIDE_VIEW"]
    assert len(unsupported) == 3
    assert all(r["method"] == "manual_roi" and r["local_circle_candidates"] is None for r in unsupported)
    assert all(r["mask_area_px"] is None and r["circle_candidates"] is None
               and r["components"] is None for r in unsupported)
    supported_zero = [r for r in report["rows"] if r["method"] == "manual_roi"
                      and r["capture_mode"] == MODES[1] and r["local_circle_candidates"] == 0]
    assert len(supported_zero) == 15
    assert all(r["status"] == "DIAGNOSTIC_ONLY" for r in supported_zero)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    assert before == after
    with (output / "features.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 360
    assert all(r["predicted_weight_g"] == "" for r in csv_rows)
    assert (output / "index.html").is_file()

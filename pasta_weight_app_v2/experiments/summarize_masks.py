"""Summarize offline masks; descriptive associations are never accuracy claims.

This file can be run directly with a separate plotting Python environment. It
does not import the application, open SQLite, train a model, or change inputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import platform

import numpy as np


MODES = (
    "spread_overlap", "bundle_ends_low_angle", "handheld_bundle_ends",
)
MODE_LABELS = {
    "spread_overlap": "A: spread / overlapping",
    "bundle_ends_low_angle": "B: low-angle bundle ends",
    "handheld_bundle_ends": "C: handheld ends / side",
}
FEATURES = (
    ("mask_area_px", "Foreground area (px)"),
    ("mask_fraction", "Foreground / image area"),
    ("long_axis_px", "Rotated box long axis (px)"),
    ("short_axis_px", "Rotated box short axis (px)"),
    ("contour_perimeter_px", "External contour perimeter (px)"),
    ("components", "Connected regions (not strands)"),
    ("circle_candidates", "Global circle candidates"),
)
MORPHOLOGY = (
    ("skeleton_length_512", "Skeleton length at max-side 512 (px)"),
    ("skeleton_width_512", "Skeleton width at max-side 512 (px)"),
    ("skeleton_density_512", "Skeleton-region density at max-side 512"),
    ("rect_density", "Foreground / rotated-box area"),
    ("local_circle_candidates", "Manual-ROI local circle candidates"),
    ("watershed_regions", "Watershed regions (not strands)"),
)
NOTE = (
    "全15枚は同一物理試料系列です。重量との相関は系列内の記述統計であり、"
    "独立精度・因果関係・重量予測性能を示しません。背景や手が特徴量に混ざり得ます。"
    "手動ROIは未承認の検品用条件です。未対応の側面画像と欠損値は集計から除外し、"
    "0として扱いません。デジタル縮小は実際の撮影距離の実験ではありません。"
)


def number(value):
    """Return a finite numeric value, preserving actual zero and rejecting bool."""
    return (float(value) if isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(value) else None)


def eligible(row):
    return row.get("status") != "UNSUPPORTED_SIDE_VIEW"


def association(rows, feature):
    pairs = [(number(r.get("nominal_weight_g")), number(r.get(feature)))
             for r in rows if eligible(r)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    result = {"feature": feature, "n_images": len(pairs), "pearson_r": None,
              "independent_series": 1, "interpretation": "dependent-series descriptive only"}
    if len(pairs) >= 2:
        x, y = np.asarray(pairs, dtype=float).T
        if np.std(x) > 0 and np.std(y) > 0:
            result["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
    return result


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["scale"] == 1 and eligible(row):
            groups[(row["capture_mode"], row["method"])].append(row)
    summary, correlations = [], []
    feature_keys = [k for k, _ in FEATURES + MORPHOLOGY]
    for (mode, method), items in sorted(groups.items()):
        for feature in feature_keys:
            values = [number(r.get(feature)) for r in items]
            values = [v for v in values if v is not None]
            summary.append(dict(capture_mode=mode, method=method, feature=feature,
                                n_images=len(values), minimum=min(values) if values else None,
                                maximum=max(values) if values else None,
                                mean=float(np.mean(values)) if values else None))
            correlations.append(dict(capture_mode=mode, method=method,
                                     **association(items, feature)))
    return summary, correlations


def scale_changes(rows):
    """Compare normalized areas and actual counts to the same image at scale 1."""
    reference = {(r["image_id"], r["method"]): r for r in rows
                 if r["scale"] == 1 and eligible(r)}
    result = []
    for row in rows:
        if not eligible(row) or (row["image_id"], row["method"]) not in reference:
            continue
        base = reference[(row["image_id"], row["method"])]
        out = {k: row[k] for k in ("image_id", "method", "capture_mode", "nominal_weight_g", "scale")}
        area, area0 = number(row.get("mask_fraction")), number(base.get("mask_fraction"))
        out["foreground_fraction_relative_change_pct"] = (
            100 * (area / area0 - 1) if area is not None and area0 not in (None, 0) else None)
        for feature in ("components", "circle_candidates", "local_circle_candidates", "legacy_detected_count"):
            value, origin = number(row.get(feature)), number(base.get(feature))
            out[feature + "_change"] = value - origin if value is not None and origin is not None else None
            out[feature + "_unchanged"] = value == origin if value is not None and origin is not None else None
        result.append(out)
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trends(rows, methods, features, output, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(features), 3, figsize=(18, 3 * len(features)), squeeze=False)
    colors = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "x", "*"]
    for column, mode in enumerate(MODES):
        for line, method in enumerate(methods):
            series = sorted((r for r in rows if r["scale"] == 1 and r["capture_mode"] == mode
                             and r["method"] == method), key=lambda r: r["nominal_weight_g"])
            x = [r["nominal_weight_g"] for r in series]
            for row_index, (feature, label) in enumerate(features):
                # Missing/unsupported rows remain gaps; do not connect across them.
                y = [number(r.get(feature)) if eligible(r) else None for r in series]
                y = [np.nan if v is None else v for v in y]
                axes[row_index, column].plot(x, y, label=method, color=colors(line % 10),
                                             marker=markers[line % len(markers)], linewidth=1.1,
                                             markersize=4, alpha=.85)
        for row_index, (_, label) in enumerate(features):
            ax = axes[row_index, column]
            ax.set_title(MODE_LABELS[mode] if row_index == 0 else "", fontsize=12)
            ax.set_ylabel(label, fontsize=9)
            ax.set_xlabel("Nominal weight (g); same physical series", fontsize=8)
            ax.set_xticks([20, 40, 60, 80, 100])
            ax.grid(alpha=.25)
            if all(not np.isfinite(line.get_ydata()).any() for line in ax.lines):
                ax.text(.5, .5, "Not applicable / unavailable", transform=ax.transAxes,
                        ha="center", va="center", color="gray")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(.5, .98), fontsize=10)
    fig.suptitle(title + "\nOne dependent series; no accuracy claim; missing values are gaps", fontsize=15, y=.999)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output, dpi=135, facecolor="white")
    plt.close(fig)


def scale_plot(changes, methods, output):
    import matplotlib.pyplot as plt
    metrics = (
        ("foreground_fraction_relative_change_pct", "Mean absolute foreground-fraction change (%)"),
        ("components_change", "Mean absolute connected-region count change"),
        ("circle_candidates_change", "Mean absolute global-circle count change"),
        ("local_circle_candidates_change", "Mean absolute local-circle count change (manual ROI)"),
    )
    fig, axes = plt.subplots(len(metrics), 3, figsize=(18, 13), squeeze=False)
    for column, mode in enumerate(MODES):
        for line, method in enumerate(methods):
            for index, (metric, label) in enumerate(metrics):
                means = []
                for scale in (.5, .75, 1.):
                    values = [number(r.get(metric)) for r in changes if r["method"] == method
                              and r["capture_mode"] == mode and r["scale"] == scale]
                    values = [abs(v) for v in values if v is not None]
                    means.append(float(np.mean(values)) if values else np.nan)
                axes[index, column].plot([.5, .75, 1.], means, marker="o", label=method,
                                         color=plt.get_cmap("tab10")(line % 10), alpha=.8)
        for index, (_, label) in enumerate(metrics):
            ax = axes[index, column]
            ax.set_title(MODE_LABELS[mode] if index == 0 else "")
            ax.set_ylabel(label, fontsize=8)
            ax.set_xlabel("Digital image scale (not camera distance)", fontsize=8)
            ax.set_xticks([.5, .75, 1.]); ax.grid(alpha=.25)
            if all(not np.isfinite(line.get_ydata()).any() for line in ax.lines):
                ax.text(.5, .5, "Not applicable / unavailable", transform=ax.transAxes,
                        ha="center", color="gray")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(.5, .97))
    fig.suptitle("Resize sensitivity vs same image at scale 1\nCounts are candidates, not verified strands", y=.999)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(output, dpi=135, facecolor="white"); plt.close(fig)


def value_text(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.5g}"
    return html.escape(str(value))


def write_html(output, rows, methods, correlations):
    columns = [("image_id", "画像"), ("nominal_weight_g", "名目g"), ("capture_mode", "状態"),
               ("method", "方式"), ("scale", "縮小率"), ("status", "処理状態")] + list(FEATURES + MORPHOLOGY)
    rendered = []
    for row in rows:
        attrs = f'data-mode="{html.escape(row["capture_mode"])}" data-method="{html.escape(row["method"])}" data-scale="{row["scale"]}"'
        cells = []
        for key, _ in columns:
            v = row.get(key)
            if not eligible(row) and key in dict(FEATURES + MORPHOLOGY):
                v = None
            cells.append("<td>" + value_text(v) + "</td>")
        rendered.append("<tr " + attrs + ">" + "".join(cells) + "</tr>")
    filters = ""
    for key, label, choices in (("mode", "撮影状態", MODES), ("method", "方式", methods),
                                 ("scale", "画像サイズ", [1., .75, .5])):
        filters += f'<label>{label} <select id="{key}"><option value="all">すべて</option>'
        filters += "".join(f'<option value="{html.escape(str(c))}">{html.escape(str(c))}</option>' for c in choices)
        filters += "</select></label> "
    page = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>マスク特徴量・縮小比較（非公開）</title><style>
body{font-family:sans-serif;color:#29352e;background:#faf8f2;margin:20px}img{max-width:100%;height:auto}select{padding:8px;margin:6px;max-width:100%}
.table{overflow:auto;max-height:70vh}table{border-collapse:collapse;background:white;font-size:13px}td,th{padding:8px;border:1px solid #ddd;white-space:nowrap}th{position:sticky;top:0;background:#e6e8df}
article{padding:16px;background:white;border-radius:12px;margin:18px 0}a{color:#36573e}h1{font-size:25px}</style>
<h1>マスク特徴量と画像縮小の比較</h1>'''
    page += "<p>" + NOTE + "</p><p>実重量予測・MAE・独立評価・モデル学習は実施していません。— は未対応／未取得です。0とは異なります。</p>"
    page += '<p><a href="summary.csv">方式別集計CSV</a> · <a href="correlations.csv">相関CSV（記述統計のみ）</a> · <a href="scale_changes.csv">画像別縮小変化CSV</a> · <a href="summary.json">再現情報JSON</a></p>'
    page += "<article><h2>全条件の特徴量</h2>" + filters + '<p id="visible"></p><div class="table"><table><thead><tr>'
    page += "".join("<th>" + html.escape(label) + "</th>" for _, label in columns)
    page += "</tr></thead><tbody>" + "".join(rendered) + "</tbody></table></div></article>"
    for file, title in (("feature_trends.png", "重量増加と特徴量"), ("morphology_trends.png", "骨格・密度・分割候補"),
                        ("scale_sensitivity.png", "デジタル縮小の影響")):
        page += f'<article><h2>{title}</h2><a href="{file}"><img loading="lazy" src="{file}" alt="{title}"></a></article>'
    page += '''<script>
const keys=['mode','method','scale'];function filter(){let n=0;document.querySelectorAll('tbody tr').forEach(r=>{r.hidden=keys.some(k=>{const v=document.getElementById(k).value;return v!=='all'&&r.dataset[k]!==v});if(!r.hidden)n++});document.getElementById('visible').textContent=n+' 条件を表示';}
keys.forEach(k=>document.getElementById(k).addEventListener('change',filter));document.getElementById('scale').value='1.0';filter();</script></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")


def run(input_path, output):
    input_path, output = Path(input_path), Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing report")
    raw = input_path.read_bytes(); source = json.loads(raw)
    if (source.get("families") != 1 or source.get("training_images") != 0
            or source.get("independent_test_images") != 0):
        raise ValueError("This report is only for an untrained single-series comparison")
    rows, methods = source["rows"], source["methods"]
    if not rows or any(r.get("capture_mode") not in MODES for r in rows):
        raise ValueError("Missing rows or unsupported capture state")
    summary, correlations = aggregate(rows)
    changes = scale_changes(rows)
    import matplotlib
    matplotlib.use("Agg")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "correlations.csv", correlations)
    write_csv(output / "scale_changes.csv", changes)
    trends(rows, methods, FEATURES, output / "feature_trends.png", "Mask features vs nominal weight")
    trends(rows, methods, MORPHOLOGY, output / "morphology_trends.png", "Morphology features vs nominal weight")
    scale_plot(changes, methods, output / "scale_sensitivity.png")
    report = dict(purpose="descriptive_offline_comparison", independent_series=1,
                  independent_evaluation_images=0, training_images=0, accuracy_metrics=None,
                  note=NOTE, input_sha256=hashlib.sha256(raw).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  versions=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
                  excluded_unsupported_rows=sum(not eligible(r) for r in rows),
                  summary=summary, correlations=correlations, scale_changes=changes)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_html(output, rows, methods, correlations)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    run(arguments.input, arguments.output)

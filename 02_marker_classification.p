"""
Script 2: ROI curation in Fiji + marker classification.

Opens each image in ONE persistent Fiji session, lets you delete false ROIs,
asks for 5 clearly positive RBPMS ROIs per marker, re-measures the edited ROIs
directly from the TIFF, sets a per-marker/per-image threshold, and exports
single / double / triple positive populations as tables and Fiji ROI ZIPs.

Usage:
    python 02_marker_classification.py --input-dir "D:/retina/images"
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from roifile import roiread

from common import (
    CLASSIFIABLE_MARKERS,
    RBPMS_NAME,
    convert_to_zcyx,
    find_source_tiffs,
    mean_projection_z,
    parse_marker_order,
    quantify_rois,
    read_tiff_with_axes,
)

# ============================================================
# SETTINGS
# ============================================================

METRIC_SUFFIX = "corrected_p95"
REQUIRED_POSITIVE_ANCHORS = 5
REUSE_SAVED_ANCHOR_IDS = True

EXPORT_SINGLE_POSITIVE_ROIS = True
EXPORT_AT_LEAST_DOUBLE_TRIPLE = True
EXPORT_EXACT_DOUBLE_TRIPLE = True
EXPORT_EMPTY_ZIPS = False


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_filename(text):
    text = str(text).replace("+", "_PLUS_")
    return re.sub(r"[^a-zA-Z0-9_.\-]+", "_", text)


def metric_name(marker):
    return f"{marker}_{METRIC_SUFFIX}"


def image_marker_key(image_name, marker):
    return f"{image_name}|||{marker}"


def load_calibration(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_calibration(calibration, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ============================================================
# TIFF -> PROJECTIONS / REVIEW IMAGE
# ============================================================

def load_projections(image_path):
    image, axes = read_tiff_with_axes(image_path)
    zcyx = convert_to_zcyx(image, axes)
    z_count, n_channels = int(zcyx.shape[0]), int(zcyx.shape[1])
    marker_order, _ = parse_marker_order(image_path.name, n_channels, require_rbpms=False)
    projections = mean_projection_z(zcyx)
    channel_map = {m: i + 1 for i, m in enumerate(marker_order)}
    return projections, marker_order, channel_map, z_count


def prepare_review_image(image_path, projections, marker_order):
    """Small mean-Z multichannel image for fast review in Fiji."""
    review_path = image_path.with_name(image_path.stem + "_SCRIPT2_review_mean.tif")
    if (not review_path.exists()
            or review_path.stat().st_mtime < image_path.stat().st_mtime):
        tifffile.imwrite(
            review_path,
            projections.astype(np.float32, copy=False),
            imagej=True,
            metadata={"axes": "CYX", "Labels": marker_order},
        )
    return review_path


# ============================================================
# IMAGEJ ROI READING / RASTERIZATION
# ============================================================

def roi_id_from_name(name):
    if name is None:
        return None
    stem = Path(str(name)).stem
    match = re.fullmatch(r"(?:cell[_\-]?)?0*(\d+)", stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def load_curated_rois(roi_zip):
    """Read the CURRENT Fiji ROI ZIP. Names must stay Cell_XXXXX; shapes may be edited."""
    if not roi_zip.exists():
        raise FileNotFoundError(f"ROI ZIP not found:\n{roi_zip}")

    rois = roiread(roi_zip)
    if not isinstance(rois, list):
        rois = [rois]

    result, unrecognized = {}, []
    for roi in rois:
        roi_id = roi_id_from_name(roi.name)
        if roi_id is None:
            unrecognized.append(str(roi.name))
        else:
            result[roi_id] = roi

    if not result:
        raise ValueError(
            f"No Cell_XXXXX ROIs could be read from:\n{roi_zip}\n"
            "You may delete or edit ROI shapes in Fiji, but do not rename ROIs."
        )
    return result, unrecognized


def roi_to_mask(roi, height, width):
    """One ImageJ polygon/freehand ROI -> boolean mask."""
    coords = np.asarray(roi.coordinates(), dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] < 3 or coords.shape[1] != 2:
        raise ValueError(f"ROI {roi.name} is not a usable polygon/freehand ROI.")

    points = np.rint(coords).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def quantify_current_rois(image_path, roi_zip, projections, marker_order, z_count):
    """Measure raw mean projections with the CURRENT (edited) ROI shapes."""
    rois, unrecognized = load_curated_rois(roi_zip)
    height, width = int(projections.shape[1]), int(projections.shape[2])
    roi_ids = sorted(rois)

    all_cells = np.zeros((height, width), dtype=bool)
    for roi_id in roi_ids:
        all_cells |= roi_to_mask(rois[roi_id], height, width)

    df = quantify_rois(
        image_path=image_path,
        roi_ids=roi_ids,
        get_mask=lambda i: roi_to_mask(rois[i], height, width),
        all_cells=all_cells,
        projections=projections,
        marker_order=marker_order,
        z_count=z_count,
        extra={"roi_zip": str(roi_zip)},
    )
    return df, unrecognized


# ============================================================
# FIJI EXECUTABLE
# ============================================================

def find_fiji_executable(fiji_exe_arg, path_file):
    if fiji_exe_arg:
        candidate = Path(fiji_exe_arg)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"--fiji-exe does not exist:\n{candidate}")

    if path_file.exists():
        remembered = path_file.read_text(encoding="utf-8").strip().strip('"')
        if remembered and Path(remembered).is_file():
            return Path(remembered)

    home = Path.home()
    candidates = [
        Path(r"C:\Fiji.app\ImageJ-win64.exe"),
        Path(r"C:\Program Files\Fiji.app\ImageJ-win64.exe"),
        Path(r"C:\Program Files (x86)\Fiji.app\ImageJ-win64.exe"),
        home / "Fiji.app" / "ImageJ-win64.exe",
        home / "Desktop" / "Fiji.app" / "ImageJ-win64.exe",
        home / "Downloads" / "Fiji.app" / "ImageJ-win64.exe",
        Path("/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"),
        home / "Applications" / "Fiji.app" / "Contents" / "MacOS" / "ImageJ-macosx",
        Path("/opt/Fiji.app/ImageJ-linux64"),
        home / "Fiji.app" / "ImageJ-linux64",
    ]
    for candidate in candidates:
        if candidate.is_file():
            path_file.write_text(str(candidate), encoding="utf-8")
            return candidate

    for exe_name in ("ImageJ-win64.exe", "ImageJ.exe", "fiji.exe", "fiji"):
        found = shutil.which(exe_name)
        if found:
            path_file.write_text(found, encoding="utf-8")
            return Path(found)

    print("\nFiji/ImageJ was not found automatically.")
    print(r"Example: C:\Fiji.app\ImageJ-win64.exe")
    entered = input("\nPaste the full Fiji/ImageJ executable path: ").strip().strip('"')
    candidate = Path(entered)
    if not candidate.is_file():
        raise FileNotFoundError(f"That executable does not exist:\n{candidate}")
    path_file.write_text(str(candidate), encoding="utf-8")
    return candidate


def ij_macro_path(path):
    return str(path).replace("\\", "/").replace('"', '\\"')


# ============================================================
# ONE PERSISTENT FIJI SESSION
# ============================================================

def start_persistent_fiji(fiji_exe):
    """
    Start Fiji ONCE. Python and Fiji talk through tiny flag files in a temp
    directory, so the JVM is not restarted for every image.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="rbpms_fiji_batch_"))
    macro_path = temp_dir / "persistent_review.ijm"
    load_flag = temp_dir / "load.flag"
    done_flag = temp_dir / "done.flag"
    ready_flag = temp_dir / "ready.flag"
    stop_flag = temp_dir / "stop.flag"
    image_txt = temp_dir / "image.txt"
    roi_txt = temp_dir / "roi.txt"
    output_txt = temp_dir / "output.txt"

    macro = (
        f'loadFlag = "{ij_macro_path(load_flag)}";\n'
        f'doneFlag = "{ij_macro_path(done_flag)}";\n'
        f'readyFlag = "{ij_macro_path(ready_flag)}";\n'
        f'stopFlag = "{ij_macro_path(stop_flag)}";\n'
        f'imageTxt = "{ij_macro_path(image_txt)}";\n'
        f'roiTxt = "{ij_macro_path(roi_txt)}";\n'
        f'outputTxt = "{ij_macro_path(output_txt)}";\n\n'
        'File.saveString("ready", readyFlag);\n'
        'while (File.exists(stopFlag) == 0) {\n'
        '    if (File.exists(loadFlag)) {\n'
        '        imagePath = File.openAsString(imageTxt);\n'
        '        roiPath = File.openAsString(roiTxt);\n'
        '        outputPath = File.openAsString(outputTxt);\n'
        '        imagePath = replace(imagePath, "\\r", "");\n'
        '        imagePath = replace(imagePath, "\\n", "");\n'
        '        roiPath = replace(roiPath, "\\r", "");\n'
        '        roiPath = replace(roiPath, "\\n", "");\n'
        '        outputPath = replace(outputPath, "\\r", "");\n'
        '        outputPath = replace(outputPath, "\\n", "");\n'
        '        File.delete(loadFlag);\n'
        '        if (File.exists(readyFlag)) File.delete(readyFlag);\n'
        '        roiManager("reset");\n'
        '        if (File.exists(imagePath) == 0) {\n'
        '            showMessage("Script 2 error", "Review image not found:\\n" + imagePath);\n'
        '        } else if (File.exists(roiPath) == 0) {\n'
        '            showMessage("Script 2 error", "ROI ZIP not found:\\n" + roiPath);\n'
        '        } else {\n'
        '            open(imagePath);\n'
        '            roiManager("Open", roiPath);\n'
        '            roiManager("Show All with labels");\n'
        '            run("Brightness/Contrast...");\n'
        '            showStatus("RBPMS review active - delete false ROIs, inspect marker channels, then answer Python.");\n'
        '        }\n'
        '        while (File.exists(doneFlag) == 0 && File.exists(stopFlag) == 0) {\n'
        '            wait(200);\n'
        '        }\n'
        '        if (File.exists(doneFlag)) {\n'
        '            File.delete(doneFlag);\n'
        '            roiManager("Deselect");\n'
        '            roiManager("Save", outputPath);\n'
        '            close();\n'
        '            roiManager("reset");\n'
        '            File.saveString("ready", readyFlag);\n'
        '        }\n'
        '    }\n'
        '    wait(200);\n'
        '}\n'
        'if (nImages > 0) close();\n'
        'roiManager("reset");\n'
        'eval("script", "System.exit(0);");\n'
    )
    macro_path.write_text(macro, encoding="utf-8")

    print("\nStarting Fiji once for the entire batch...")
    process = subprocess.Popen([str(fiji_exe), "-port0", "-macro", str(macro_path)])

    deadline = time.time() + 120
    while not ready_flag.exists() and time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Fiji closed during startup.")
        time.sleep(0.25)
    if not ready_flag.exists():
        raise RuntimeError("Fiji did not become ready within 120 seconds.")

    return {
        "process": process, "temp_dir": temp_dir,
        "load_flag": load_flag, "done_flag": done_flag,
        "ready_flag": ready_flag, "stop_flag": stop_flag,
        "image_txt": image_txt, "roi_txt": roi_txt, "output_txt": output_txt,
    }


def _wait_for(condition, timeout_s, interval=0.2):
    deadline = time.time() + timeout_s
    while not condition() and time.time() < deadline:
        time.sleep(interval)
    return condition()


def open_image_in_persistent_fiji(session, review_image, roi_zip):
    if not _wait_for(session["ready_flag"].exists, 120):
        raise RuntimeError("Fiji was not ready for the next image.")

    temp_curated_zip = session["temp_dir"] / "curated_current.zip"
    if temp_curated_zip.exists():
        temp_curated_zip.unlink()

    session["image_txt"].write_text(ij_macro_path(review_image), encoding="utf-8")
    session["roi_txt"].write_text(ij_macro_path(roi_zip), encoding="utf-8")
    session["output_txt"].write_text(ij_macro_path(temp_curated_zip), encoding="utf-8")
    session["load_flag"].write_text("load", encoding="utf-8")

    # Fiji deletes the load flag once it has consumed the request.
    if not _wait_for(lambda: not session["load_flag"].exists(), 120):
        raise RuntimeError("Fiji did not load the requested image within 120 seconds.")

    return temp_curated_zip


def finish_current_fiji_image(session, original_roi_zip, temp_curated_zip):
    """Fiji saves the CURRENT ROI Manager, closes this image, stays open."""
    session["done_flag"].write_text("done", encoding="utf-8")

    if not _wait_for(session["ready_flag"].exists, 120):
        raise RuntimeError("Fiji did not finish saving the current image within 120 seconds.")

    if not temp_curated_zip.exists():
        raise RuntimeError(
            "Fiji did not produce the curated ROI ZIP. The original ROI ZIP was not replaced."
        )

    tmp_dest = original_roi_zip.with_name(original_roi_zip.stem + "_saving_tmp.zip")
    if tmp_dest.exists():
        tmp_dest.unlink()
    shutil.copy2(temp_curated_zip, tmp_dest)
    os.replace(tmp_dest, original_roi_zip)
    temp_curated_zip.unlink(missing_ok=True)


def stop_persistent_fiji(session):
    if session is None:
        return
    try:
        session["stop_flag"].write_text("stop", encoding="utf-8")
        session["process"].wait(timeout=30)
    except Exception:
        try:
            session["process"].terminate()
        except Exception:
            pass
    finally:
        shutil.rmtree(session["temp_dir"], ignore_errors=True)


# ============================================================
# USER POSITIVE ANCHORS
# ============================================================

def parse_roi_ids(text):
    roi_ids = []
    for token in (t.strip() for t in re.split(r"[,;\s]+", text) if t.strip()):
        t = token.lower()
        match = re.fullmatch(r"cell[_\-]?0*(\d+)", t)
        if match:
            roi_id = int(match.group(1))
        elif t.isdigit():
            roi_id = int(t)
        else:
            raise ValueError(f"Could not understand ROI '{token}'.")
        if roi_id not in roi_ids:
            roi_ids.append(roi_id)
    return roi_ids


def ask_for_anchor_ids(image_name, marker, current_roi_ids, channel_number,
                       old_info=None, recalibrate=False):
    """Ask for the 5 positive ROI IDs while Fiji is open. Thresholds come later."""
    if (old_info
            and old_info.get("status") == "active"
            and REUSE_SAVED_ANCHOR_IDS
            and not recalibrate):
        old_ids = [int(x) for x in old_info.get("anchor_roi_ids", [])]
        if (len(old_ids) == REQUIRED_POSITIVE_ANCHORS
                and all(i in current_roi_ids for i in old_ids)):
            print(f"\n{marker.upper()}: reusing your saved positive ROI IDs {old_ids}")
            return old_ids

    print("\n" + "=" * 100)
    print(f"IMAGE  : {image_name}")
    print(f"MARKER : {marker.upper()}")
    print(f"FIJI CHANNEL: {channel_number}")
    print("=" * 100)
    print("In the open Fiji image, switch to this channel and choose "
          f"{REQUIRED_POSITIVE_ANCHORS} RBPMS ROIs that YOU consider positive.")
    print("Try to include at least one weak-but-real positive.")
    print("\nExample: 865, 5, 1220, 11, 482")
    print("Type 'skip' to skip this marker.")

    while True:
        answer = input(
            f"\nEnter {REQUIRED_POSITIVE_ANCHORS} positive ROI IDs for {marker}, or 'skip': "
        ).strip()

        if answer.lower() == "skip":
            return None

        try:
            roi_ids = parse_roi_ids(answer)
        except ValueError as exc:
            print(f"Input problem: {exc}")
            continue

        if len(roi_ids) != REQUIRED_POSITIVE_ANCHORS:
            print(f"Enter exactly {REQUIRED_POSITIVE_ANCHORS} different IDs.")
            continue

        missing = [i for i in roi_ids if i not in current_roi_ids]
        if missing:
            print(f"These ROI IDs are not in the current ROI ZIP: {missing}")
            continue

        return roi_ids


# ============================================================
# CLASSIFICATION
# ============================================================

def calculate_threshold_from_anchors(measurements, marker, anchor_ids):
    metric = metric_name(marker)
    selected = measurements[measurements["cell_id"].astype(int).isin(anchor_ids)].copy()

    if len(selected) != len(anchor_ids):
        available = set(selected["cell_id"].astype(int))
        missing = [i for i in anchor_ids if i not in available]
        raise ValueError(
            f"{marker}: anchor ROI(s) disappeared during Fiji curation: {missing}. "
            "Re-run this image and choose replacement anchors."
        )

    values = pd.to_numeric(selected[metric], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{marker}: an anchor ROI has no usable {metric} value.")

    threshold = float(values.min())
    scores = {str(int(r["cell_id"])): float(r[metric]) for _, r in selected.iterrows()}
    return threshold, scores


def apply_marker_classification(measurements, marker, threshold):
    values = pd.to_numeric(measurements[metric_name(marker)], errors="coerce")
    positive = pd.Series(pd.NA, index=measurements.index, dtype="boolean")
    valid = values.notna()
    positive.loc[valid] = values.loc[valid] >= threshold

    measurements[f"{marker}_positive"] = positive
    measurements[f"{marker}_classification_metric"] = values
    measurements[f"{marker}_classification_threshold"] = threshold
    return measurements


# ============================================================
# ROI ZIP SUBSETTING
# ============================================================

def export_roi_subset(source_roi_zip, roi_ids, output_zip):
    wanted = {int(x) for x in roi_ids}
    selected = []

    with zipfile.ZipFile(source_roi_zip, "r") as zin:
        for member in zin.namelist():
            if not member.lower().endswith(".roi"):
                continue
            if roi_id_from_name(Path(member).name) in wanted:
                selected.append((member, zin.read(member)))

    if output_zip.exists():
        output_zip.unlink()

    if selected or EXPORT_EMPTY_ZIPS:
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for member, data in selected:
                zout.writestr(member, data)

    return len(selected)


# ============================================================
# SUMMARY / DOUBLE / TRIPLE POSITIVES
# ============================================================

def summarize_and_export(group, active_markers, image_path, source_roi_zip, roi_output_dir):
    image_name = image_path.name
    stem = safe_filename(image_path.stem)
    image_out = roi_output_dir / stem
    image_out.mkdir(parents=True, exist_ok=True)

    def pct(n, d):
        return 100.0 * n / d if d else np.nan

    def export(ids, label, enabled):
        if enabled and (ids or EXPORT_EMPTY_ZIPS):
            export_roi_subset(source_roi_zip, ids, image_out / f"{stem}_{label}_ROIs.zip")

    def ids_of(mask):
        return group.loc[mask, "cell_id"].astype(int).tolist()

    known = {m: group[f"{m}_positive"].notna() for m in active_markers}
    is_pos = {m: group[f"{m}_positive"].fillna(False).astype(bool) for m in active_markers}

    single_rows, combo_rows = [], []

    for marker in active_markers:
        ids = ids_of(known[marker] & is_pos[marker])
        denom = int(known[marker].sum())
        single_rows.append({
            "image": image_name,
            "marker": marker,
            "eligible_rbpms_cells": denom,
            "positive_count": len(ids),
            "positive_percent": pct(len(ids), denom),
        })
        export(ids, f"{marker}_POS", EXPORT_SINGLE_POSITIVE_ROIS)

    for size in (2, 3):
        if len(active_markers) < size:
            continue

        for combo in combinations(active_markers, size):
            combo_name = "+".join(combo)

            eligible = pd.Series(True, index=group.index, dtype=bool)
            positive = pd.Series(True, index=group.index, dtype=bool)
            for m in combo:
                eligible &= known[m]
                positive &= is_pos[m]

            # at least: all markers in the combo are positive
            atleast_ids = ids_of(eligible & positive)
            n_elig = int(eligible.sum())
            combo_rows.append({
                "image": image_name, "combination": combo_name,
                "combination_size": size, "mode": "at_least",
                "eligible_rbpms_cells": n_elig,
                "positive_count": len(atleast_ids),
                "positive_percent": pct(len(atleast_ids), n_elig),
            })
            export(atleast_ids, f"ATLEAST_{safe_filename(combo_name)}",
                   EXPORT_AT_LEAST_DOUBLE_TRIPLE)

            # exact: combo positive AND every other classified marker negative
            exact_eligible = eligible.copy()
            exact = positive.copy()
            for m in active_markers:
                if m in combo:
                    continue
                exact_eligible &= known[m]
                exact &= ~is_pos[m]
            exact &= exact_eligible

            exact_ids = ids_of(exact)
            n_exact_elig = int(exact_eligible.sum())
            combo_rows.append({
                "image": image_name, "combination": combo_name,
                "combination_size": size, "mode": "exact",
                "eligible_rbpms_cells": n_exact_elig,
                "positive_count": len(exact_ids),
                "positive_percent": pct(len(exact_ids), n_exact_elig),
            })
            export(exact_ids, f"EXACT_{safe_filename(combo_name)}",
                   EXPORT_EXACT_DOUBLE_TRIPLE)

    return single_rows, combo_rows


def make_simple_report(classified, single_df, combo_df):
    total_by_image = classified.groupby("image", dropna=False).size().to_dict()
    rows = []

    if not single_df.empty:
        for _, r in single_df.iterrows():
            rows.append({
                "image": r["image"],
                "total_rbpms": int(total_by_image.get(r["image"], 0)),
                "population": f"{r['marker']}+",
                "positive_count": int(r["positive_count"]),
                "percent_of_rbpms": float(r["positive_percent"]),
            })

    if not combo_df.empty:
        for _, r in combo_df[combo_df["mode"] == "at_least"].iterrows():
            rows.append({
                "image": r["image"],
                "total_rbpms": int(total_by_image.get(r["image"], 0)),
                "population": f"{r['combination']}+",
                "positive_count": int(r["positive_count"]),
                "percent_of_rbpms": float(r["positive_percent"]),
            })

    return pd.DataFrame(
        rows,
        columns=["image", "total_rbpms", "population", "positive_count", "percent_of_rbpms"],
    )


# ============================================================
# MAIN
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Fiji ROI curation and marker classification for RBPMS+ cells."
    )
    ap.add_argument("--input-dir", type=Path, default=Path("."),
                    help="Folder with the original TIFFs and *_RBPMS_ROIs.zip files.")
    ap.add_argument("--fiji-exe", type=str, default=None,
                    help="Full path to the Fiji/ImageJ executable (auto-detected if omitted).")
    ap.add_argument("--recalibrate", action="store_true",
                    help="Ignore saved anchor ROI IDs and ask again for every marker.")
    return ap.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()

    calibration_json = input_dir / "RBPMS_positive_anchor_calibration.json"
    out_measurements = input_dir / "RBPMS_curated_measurements.csv"
    out_classified = input_dir / "RBPMS_all_cells_classified.csv"
    out_single = input_dir / "RBPMS_single_marker_summary.csv"
    out_combo = input_dir / "RBPMS_double_triple_summary.csv"
    out_report = input_dir / "RBPMS_simple_report.csv"
    out_curation = input_dir / "RBPMS_roi_curation_summary.csv"
    roi_output_dir = input_dir / "RBPMS_classified_ROIs"
    fiji_path_file = input_dir / "RBPMS_fiji_executable.txt"

    print("\n" + "=" * 100)
    print("CURATED RBPMS + MARKER CLASSIFICATION - script 2")
    print("=" * 100)
    print("\nRequired per image: original multichannel TIFF + matching *_RBPMS_ROIs.zip")
    print("Current Fiji ROI shapes are re-measured directly from the TIFF.")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist:\n{input_dir}")

    source_tiffs = find_source_tiffs(input_dir)
    if not source_tiffs:
        raise FileNotFoundError(f"No source TIFF stacks found under:\n{input_dir}")

    print(f"\nInput directory:\n{input_dir}")
    print(f"Source TIFFs found: {len(source_tiffs)}")

    calibration = load_calibration(calibration_json)

    fiji_exe = find_fiji_executable(args.fiji_exe, fiji_path_file)
    print(f"\nFiji executable:\n{fiji_exe}")
    fiji_session = start_persistent_fiji(fiji_exe)

    all_measurements, all_classified = [], []
    all_single_rows, all_combo_rows, curation_rows = [], [], []

    try:
        for image_index, image_path in enumerate(source_tiffs, start=1):
            print("\n" + "#" * 100)
            print(f"[{image_index}/{len(source_tiffs)}] {image_path.name}")
            print("#" * 100)

            roi_zip = image_path.with_name(image_path.stem + "_RBPMS_ROIs.zip")
            if not roi_zip.exists():
                print(f"\nSKIPPED: matching ROI ZIP not found:\n{roi_zip}")
                continue

            projections, marker_order, channel_map, z_count = load_projections(image_path)
            review_image = prepare_review_image(image_path, projections, marker_order)

            rois_before, _ = load_curated_rois(roi_zip)
            roi_ids_before = set(rois_before)
            count_before = len(roi_ids_before)

            print(f"\nCurrent RBPMS ROIs: {count_before}")
            print("\nChannels:")
            for marker in marker_order:
                print(f"  {channel_map[marker]} = {marker.upper()}")

            available_markers = [
                m for m in marker_order
                if m != RBPMS_NAME and m in CLASSIFIABLE_MARKERS
            ]

            temp_curated_zip = open_image_in_persistent_fiji(
                fiji_session, review_image, roi_zip
            )
            print("\nFiji is open with the mean-Z multichannel image and current RBPMS ROIs.")
            print("Delete false-positive RBPMS ROIs if needed. Do NOT rename Cell_XXXXX ROIs.")

            # Collect anchors while Fiji is open (no thresholds yet).
            anchor_ids_by_marker = {}
            for marker in available_markers:
                key = image_marker_key(image_path.name, marker)
                anchor_ids = ask_for_anchor_ids(
                    image_name=image_path.name,
                    marker=marker,
                    current_roi_ids=roi_ids_before,
                    channel_number=channel_map[marker],
                    old_info=calibration.get(key),
                    recalibrate=args.recalibrate,
                )

                if anchor_ids is None:
                    calibration[key] = {
                        "status": "skipped_by_user",
                        "image": image_path.name,
                        "marker": marker,
                    }
                    save_calibration(calibration, calibration_json)
                    continue

                anchor_ids_by_marker[marker] = anchor_ids

            # Save the edited ROI ZIP, close this image, keep Fiji open.
            finish_current_fiji_image(fiji_session, roi_zip, temp_curated_zip)

            # Re-read the FINAL edited ROIs and measure their CURRENT shapes.
            final_rois, unrecognized = load_curated_rois(roi_zip)
            final_roi_ids = set(final_rois)
            count_after = len(final_roi_ids)
            deleted_count = count_before - count_after

            print(f"\nFinal curated RBPMS count: {count_after}")
            print(f"ROIs removed during this review: {deleted_count}")
            if unrecognized:
                print(f"WARNING: {len(unrecognized)} ROI names were not recognized as Cell_XXXXX.")

            measurements, _ = quantify_current_rois(
                image_path, roi_zip, projections, marker_order, z_count
            )
            all_measurements.append(measurements.copy())

            active_markers = []

            for marker, anchor_ids in anchor_ids_by_marker.items():
                key = image_marker_key(image_path.name, marker)

                missing_anchors = [i for i in anchor_ids if i not in final_roi_ids]
                if missing_anchors:
                    print(f"\n{marker.upper()} NOT classified: anchor ROI(s) were "
                          f"deleted in Fiji: {missing_anchors}")
                    calibration[key] = {
                        "status": "needs_recalibration_anchor_deleted",
                        "image": image_path.name,
                        "marker": marker,
                        "deleted_anchor_roi_ids": missing_anchors,
                    }
                    save_calibration(calibration, calibration_json)
                    continue

                threshold, anchor_scores = calculate_threshold_from_anchors(
                    measurements, marker, anchor_ids
                )
                print(f"\n{marker.upper()} threshold from your positive ROIs:")
                print(f"  {metric_name(marker)} >= {threshold:.6g}")

                measurements = apply_marker_classification(measurements, marker, threshold)
                active_markers.append(marker)

                calibration[key] = {
                    "status": "active",
                    "image": image_path.name,
                    "marker": marker,
                    "metric": metric_name(marker),
                    "operator": ">=",
                    "threshold_last_run": threshold,
                    "anchor_roi_ids": [int(x) for x in anchor_ids],
                    "anchor_scores_last_run": anchor_scores,
                    "roi_zip_sha256_last_run": file_sha256(roi_zip),
                    "method": "minimum_score_of_5_user_confirmed_positive_rois",
                }
                save_calibration(calibration, calibration_json)

            all_classified.append(measurements.copy())

            curation_rows.append({
                "image": image_path.name,
                "rbpms_before_review": int(count_before),
                "rbpms_after_review": int(count_after),
                "deleted_false_positive_rbpms": int(deleted_count),
            })

            if active_markers:
                single_rows, combo_rows = summarize_and_export(
                    measurements, active_markers, image_path, roi_zip, roi_output_dir
                )
                all_single_rows.extend(single_rows)
                all_combo_rows.extend(combo_rows)

    finally:
        print("\nClosing Fiji...")
        stop_persistent_fiji(fiji_session)

    if not all_measurements:
        print("\nNo images were successfully quantified.")
        return

    pd.concat(all_measurements, ignore_index=True, sort=False).to_csv(
        out_measurements, index=False
    )
    classified = pd.concat(all_classified, ignore_index=True, sort=False)
    classified.to_csv(out_classified, index=False)

    single_df = pd.DataFrame(all_single_rows)
    combo_df = pd.DataFrame(all_combo_rows)
    single_df.to_csv(out_single, index=False)
    combo_df.to_csv(out_combo, index=False)
    pd.DataFrame(curation_rows).to_csv(out_curation, index=False)
    make_simple_report(classified, single_df, combo_df).to_csv(out_report, index=False)

    print("\n" + "=" * 100 + "\nFINISHED\n" + "=" * 100)
    print(f"\nFresh measurements from CURRENT edited ROIs:\n{out_measurements}")
    print(f"\nPer-cell classification:\n{out_classified}")
    print(f"\nSimple functional report:\n{out_report}")
    print(f"\nRBPMS curation summary:\n{out_curation}")
    print(f"\nDetailed single-marker counts:\n{out_single}")
    print(f"\nDetailed double/triple counts:\n{out_combo}")
    print(f"\nSeparated positive Fiji ROI ZIPs:\n{roi_output_dir}")


if __name__ == "__main__":
    main()

"""
Shared settings and helpers for the RBPMS pipeline.
Both scripts import this file, so the marker list, filename parsing, TIFF
handling, background ring and quantification are identical in both.
"""
from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd
import tifffile


# ============================================================
# MARKERS (single source of truth for both scripts)
# ============================================================

RBPMS_NAME = "rbpms"

# RGC markers only. Names must match the tokens in your filenames.
KNOWN_MARKERS = [
    "rbpms",    # pan-RGC marker
    "cart",
    "foxp2",
    "smi32",
    "spp1",
    "vglut2",
    "opn4",
    "opn4ai9",
    "tbr2",
    "sensor",   # kept: unclear what this is, delete if not RGC-related
    "cal",      # kept: calbindin/calretinin is mixed, delete if not RGC-related
]

# Markers that can be classified in script 2 (everything except RBPMS).
CLASSIFIABLE_MARKERS = [m for m in KNOWN_MARKERS if m != RBPMS_NAME]


# ============================================================
# BACKGROUND RING SETTINGS (used by both scripts)
# ============================================================

BACKGROUND_GAP_PIXELS = 3
BACKGROUND_OUTER_PIXELS = 12
MIN_BACKGROUND_PIXELS = 25
_KERNEL = np.ones((3, 3), dtype=np.uint8)


# ============================================================
# GENERATED FILE FILTERS
# ============================================================

GENERATED_TIFF_SUFFIXES = (
    "_rbpms_stack",
    "_rbpms_mean",
    "_rbpms_masks",
    "_all_channels_mean",
    "_script2_review_mean",
    "_cp_masks",
    "_cp_outlines",
    "_cp_flows",
    "_seg",
    "_masks",
    "_outlines",
    "_flows",
)


def is_generated_tiff(path: Path) -> bool:
    stem = path.stem.lower()
    return any(stem.endswith(s) for s in GENERATED_TIFF_SUFFIXES)


def find_source_tiffs(input_dir: Path):
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".tif", ".tiff"}
        and not is_generated_tiff(p)
    )


# ============================================================
# FILENAME -> CHANNEL ORDER
# ============================================================

def get_primary_filename_section(filename: str) -> str:
    """Exported names repeat the CZI name after '.czi'; use only the first part."""
    stem = Path(filename).stem
    pos = stem.lower().find(".czi")
    return stem[:pos] if pos != -1 else stem


def tokenize_filename(text: str):
    return [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if t]


def parse_marker_order(filename: str, n_channels: int, require_rbpms: bool = True):
    primary = get_primary_filename_section(filename)
    known = set(KNOWN_MARKERS)
    marker_order = [t for t in tokenize_filename(primary) if t in known]

    if require_rbpms and RBPMS_NAME not in marker_order:
        raise ValueError(
            f"RBPMS not found in filename.\nFile: {filename}\n"
            f"Primary marker section: {primary}\nDetected markers: {marker_order}"
        )

    if len(marker_order) != n_channels:
        raise ValueError(
            "CHANNEL / FILENAME MISMATCH\n"
            f"File: {filename}\nTIFF channels: {n_channels}\n"
            f"Detected markers: {len(marker_order)}\nMarker order: {marker_order}\n"
            "Every channel marker must appear in the filename, left to right. "
            "Add missing marker names to KNOWN_MARKERS in common.py."
        )

    if len(set(marker_order)) != len(marker_order):
        raise ValueError(
            f"Duplicate marker names in the primary filename section.\n"
            f"File: {filename}\nMarker order: {marker_order}"
        )

    return marker_order, primary


# ============================================================
# TIFF READING / AXES / PROJECTION
# ============================================================

def read_tiff_with_axes(path: Path):
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        return series.asarray(), series.axes


def convert_to_zcyx(image, axes):
    axes = axes.upper()
    if axes == "YX":
        return image[None, None, :, :]
    if axes == "ZYX":
        return image[:, None, :, :]
    if axes == "CYX":
        return image[None, :, :, :]
    if axes == "ZCYX":
        return image
    if axes == "CZYX":
        return np.transpose(image, (1, 0, 2, 3))
    if axes == "YXC":
        return np.transpose(image, (2, 0, 1))[None, :, :, :]
    raise ValueError(f"Unsupported TIFF layout: axes={axes}, shape={image.shape}")


def mean_projection_z(zcyx):
    """Z,C,Y,X -> C,Y,X (float32 arithmetic mean across Z)."""
    return np.mean(zcyx.astype(np.float32), axis=0, dtype=np.float32)


# ============================================================
# BACKGROUND + QUANTIFICATION
# ============================================================

def local_background_mask(roi_mask, all_cells):
    """Ring around one ROI, excluding every other cell."""
    u8 = roi_mask.astype(np.uint8)
    inner = cv2.dilate(u8, _KERNEL, iterations=BACKGROUND_GAP_PIXELS).astype(bool)
    outer = cv2.dilate(u8, _KERNEL, iterations=BACKGROUND_OUTER_PIXELS).astype(bool)
    return outer & ~inner & ~all_cells


def _safe(fn, values, *args):
    return float(fn(values, *args)) if values.size else np.nan


def quantify_rois(
    image_path,
    roi_ids,
    get_mask,
    all_cells,
    projections,
    marker_order,
    z_count,
    extra=None,
):
    """
    Measure every ROI on the raw mean-Z projections.

    roi_ids  : iterable of integer cell IDs
    get_mask : function(roi_id) -> boolean (Y,X) mask
    all_cells: boolean union of all ROIs (excluded from background)
    extra    : dict of additional columns added to every row
    """
    global_background = ~all_cells
    eps = 1e-6
    rows = []

    for roi_id in roi_ids:
        roi = get_mask(roi_id)
        area = int(roi.sum())
        if area == 0:
            continue

        ys, xs = np.where(roi)
        bg_mask = local_background_mask(roi, all_cells)
        n_local_bg = int(bg_mask.sum())

        row = {
            "image": image_path.name,
            "image_path": str(image_path),
            **(extra or {}),
            "cell_id": int(roi_id),
            "roi_name": f"Cell_{int(roi_id):05d}",
            "area_px": area,
            "centroid_x": float(xs.mean()),
            "centroid_y": float(ys.mean()),
            "z_slices": int(z_count),
            "channel_count": int(len(marker_order)),
            "markers_present": "|".join(marker_order),
            "local_background_pixels": n_local_bg,
        }

        for c, marker in enumerate(marker_order):
            img = projections[c]
            roi_values = img[roi]

            if n_local_bg >= MIN_BACKGROUND_PIXELS:
                bg_values, bg_source = img[bg_mask], "local"
            else:
                bg_values, bg_source = img[global_background], "global_fallback"

            roi_mean = _safe(np.mean, roi_values)
            roi_median = _safe(np.median, roi_values)
            roi_p90 = _safe(np.percentile, roi_values, 90)
            roi_p95 = _safe(np.percentile, roi_values, 95)
            roi_max = _safe(np.max, roi_values)

            bg_mean = _safe(np.mean, bg_values)
            bg_median = _safe(np.median, bg_values)
            bg_std = _safe(np.std, bg_values)

            s2b = (
                roi_mean / (bg_median + eps)
                if np.isfinite(roi_mean) and np.isfinite(bg_median)
                else np.nan
            )
            zscore = (
                (roi_mean - bg_mean) / bg_std
                if np.isfinite(bg_std) and bg_std > 0
                else np.nan
            )

            p = marker.lower()
            row[f"{p}_mean"] = roi_mean
            row[f"{p}_median"] = roi_median
            row[f"{p}_p90"] = roi_p90
            row[f"{p}_p95"] = roi_p95
            row[f"{p}_max"] = roi_max
            row[f"{p}_background_source"] = bg_source
            row[f"{p}_background_mean"] = bg_mean
            row[f"{p}_background_median"] = bg_median
            row[f"{p}_background_std"] = bg_std
            row[f"{p}_corrected_mean"] = roi_mean - bg_median
            row[f"{p}_corrected_median"] = roi_median - bg_median
            row[f"{p}_corrected_p90"] = roi_p90 - bg_median
            row[f"{p}_corrected_p95"] = roi_p95 - bg_median
            row[f"{p}_signal_to_background"] = s2b
            row[f"{p}_background_zscore"] = zscore

        rows.append(row)

    return pd.DataFrame(rows)

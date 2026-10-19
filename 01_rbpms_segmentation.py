"""
Script 1: RBPMS segmentation.

Mean-Z projection -> Cellpose (cpdino) on the RBPMS channel -> masks,
Fiji ROI ZIP, QC overlay and per-cell marker measurements.

Usage:
    python 01_rbpms_segmentation.py --input-dir "D:/retina/images"
"""
import argparse
import json
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from cellpose import models, utils
from roifile import ImagejRoi, roiwrite

from common import (
    BACKGROUND_GAP_PIXELS,
    BACKGROUND_OUTER_PIXELS,
    KNOWN_MARKERS,
    MIN_BACKGROUND_PIXELS,
    RBPMS_NAME,
    convert_to_zcyx,
    find_source_tiffs,
    is_generated_tiff,
    mean_projection_z,
    parse_marker_order,
    quantify_rois,
    read_tiff_with_axes,
)

# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "cpdino"
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
MIN_SIZE = 15
NORMALIZE = {"normalize": True, "percentile": [1.0, 99.0]}
TILE_SIZE = 384
BATCH_SIZE = 4
TILE_OVERLAP = 0.1
RESAMPLE = True

SAVE_RBPMS_STACK = True
SAVE_RBPMS_MEAN = True
SAVE_ALL_CHANNEL_MEANS = True
SAVE_MASK_TIFF = True
SAVE_CELLPOSE_NPY = True
SAVE_IMAGEJ_ROI_ZIP = True
SAVE_QC_OVERLAY = True
SAVE_PER_IMAGE_CSV = True


# ============================================================
# OUTPUT HELPERS
# ============================================================

def save_float_as_uint16(image, output_path, axes):
    """Display-only rescaling. Quantification never uses this copy."""
    arr = np.asarray(image, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        out = np.zeros(arr.shape, dtype=np.uint16)
    else:
        out = np.round(np.clip((arr - lo) / (hi - lo), 0, 1) * 65535).astype(np.uint16)
    tifffile.imwrite(output_path, out, metadata={"axes": axes})


def save_all_channel_means_for_fiji(projections_cyx, marker_order, output_path):
    """RAW float32 mean projections as an ImageJ C,Y,X stack (still quantitative)."""
    tifffile.imwrite(
        output_path,
        projections_cyx.astype(np.float32, copy=False),
        imagej=True,
        metadata={"axes": "CYX", "Labels": [str(m) for m in marker_order]},
    )


def save_channel_map(marker_order, output_path):
    pd.DataFrame({
        "channel_1based": np.arange(1, len(marker_order) + 1),
        "marker": marker_order,
    }).to_csv(output_path, index=False)


def save_overlay(image, masks, output_path):
    image = np.asarray(image, dtype=np.float32)
    p1, p99 = np.percentile(image, [1, 99])
    display = np.clip((image - p1) / max(p99 - p1, 1e-6), 0, 1)
    outlines = utils.masks_to_outlines(masks)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(display, cmap="gray")
    ax.imshow(np.ma.masked_where(~outlines, outlines), alpha=0.9)
    ax.set_title(f"RBPMS - {MODEL_NAME} - {int(masks.max())} ROIs")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_cellpose_segmentation(output_path, source_path, masks, flows):
    data = {
        "filename": str(source_path),
        "masks": masks,
        "outlines": utils.masks_to_outlines(masks),
        "chan_choose": [0, 0],
        "ismanual": np.zeros(int(masks.max()) + 1, dtype=bool),
        "flows": flows,
        "zdraw": [],
        "model_path": MODEL_NAME,
        "flow_threshold": FLOW_THRESHOLD,
        "cellprob_threshold": CELLPROB_THRESHOLD,
    }
    np.save(output_path, data, allow_pickle=True)


# ============================================================
# IMAGEJ / FIJI ROI ZIP
# ============================================================

def mask_to_imagej_rois(masks):
    """
    One ImageJ freehand ROI per Cellpose object.
    ROI names match CSV cell_id: Cell_00001 <-> cell_id 1.
    If an object has several contours, the largest is exported.
    """
    rois, warnings = [], []

    for roi_id in range(1, int(masks.max()) + 1):
        binary = (masks == roi_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            warnings.append(f"Cell_{roi_id:05d}: no contour found")
            continue
        if len(contours) > 1:
            warnings.append(f"Cell_{roi_id:05d}: {len(contours)} contours; largest exported")

        contour = max(contours, key=cv2.contourArea)
        points = contour[:, 0, :].astype(np.float32)  # X,Y

        if len(points) < 3:
            warnings.append(f"Cell_{roi_id:05d}: contour had fewer than 3 points")
            continue

        rois.append(ImagejRoi.frompoints(points, name=f"Cell_{roi_id:05d}"))

    return rois, warnings


def save_imagej_roi_zip(masks, output_path):
    rois, warnings = mask_to_imagej_rois(masks)
    if output_path.exists():
        output_path.unlink()
    roiwrite(output_path, rois, mode="w")
    return len(rois), warnings


# ============================================================
# SEGMENTATION
# ============================================================

def segment_rbpms(model, rbpms_mean):
    masks, flows, styles = model.eval(
        rbpms_mean,
        batch_size=BATCH_SIZE,
        normalize=NORMALIZE,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        min_size=MIN_SIZE,
        tile_overlap=TILE_OVERLAP,
        resample=RESAMPLE,
        do_3D=False,
        bsize=TILE_SIZE,
    )
    return np.asarray(masks, dtype=np.uint32), flows, styles


# ============================================================
# PROCESS ONE SOURCE TIFF
# ============================================================

def process_one(path: Path, model, reuse_masks: bool):
    print("\n" + "=" * 88)
    print(f"PROCESSING:\n{path}")
    print("=" * 88)

    image, axes = read_tiff_with_axes(path)
    zcyx = convert_to_zcyx(image, axes)
    z_count, channel_count, height, width = zcyx.shape

    marker_order, primary = parse_marker_order(path.name, channel_count)

    print(f"Original shape: {image.shape}   TIFF axes: {axes}")
    print(f"Interpreted: Z={z_count}, C={channel_count}, Y={height}, X={width}")
    print(f"Filename marker section: {primary}")
    print(f"Channel map: {marker_order}")

    rbpms_channel = marker_order.index(RBPMS_NAME)
    print(f"RBPMS channel: {rbpms_channel + 1}")

    projections = mean_projection_z(zcyx)
    rbpms_stack = zcyx[:, rbpms_channel, :, :]
    rbpms_mean = projections[rbpms_channel]

    stem = path.stem
    stack_path = path.with_name(stem + "_RBPMS_stack.tif")
    rbpms_mean_path = path.with_name(stem + "_RBPMS_mean.tif")
    all_mean_path = path.with_name(stem + "_all_channels_mean.tif")
    channel_map_path = path.with_name(stem + "_channel_map.csv")
    mask_path = path.with_name(stem + "_RBPMS_masks.tif")
    seg_path = path.with_name(stem + "_RBPMS_seg.npy")
    roi_zip_path = path.with_name(stem + "_RBPMS_ROIs.zip")
    overlay_path = path.with_name(stem + "_RBPMS_overlay.png")
    measurement_path = path.with_name(stem + "_RBPMS_marker_measurements.csv")

    if SAVE_RBPMS_STACK:
        tifffile.imwrite(stack_path, rbpms_stack, metadata={"axes": "ZYX"})
    if SAVE_RBPMS_MEAN:
        save_float_as_uint16(rbpms_mean, rbpms_mean_path, "YX")
    if SAVE_ALL_CHANNEL_MEANS:
        save_all_channel_means_for_fiji(projections, marker_order, all_mean_path)
        save_channel_map(marker_order, channel_map_path)

    reused_mask = False
    if reuse_masks and mask_path.exists():
        print("Using existing RBPMS mask; Cellpose will NOT rerun.")
        masks = tifffile.imread(mask_path).astype(np.uint32, copy=False)
        reused_mask = True
        if masks.shape != (height, width):
            raise ValueError(
                f"Existing mask size mismatch. Mask={masks.shape}, image={(height, width)}"
            )
    else:
        print(f"Running {MODEL_NAME} segmentation...")
        masks, flows, _ = segment_rbpms(model, rbpms_mean)
        if SAVE_MASK_TIFF:
            tifffile.imwrite(mask_path, masks, metadata={"axes": "YX"})
        if SAVE_CELLPOSE_NPY:
            save_cellpose_segmentation(seg_path, path, masks, flows)

    roi_count = int(masks.max())
    print(f"RBPMS ROIs: {roi_count}")

    roi_zip_count, roi_warnings = np.nan, []
    if SAVE_IMAGEJ_ROI_ZIP:
        roi_zip_count, roi_warnings = save_imagej_roi_zip(masks, roi_zip_path)
        print(f"ImageJ/Fiji ROIs exported: {roi_zip_count}")
        for msg in roi_warnings[:10]:
            print(f"  WARNING: {msg}")
        if len(roi_warnings) > 10:
            print("  ...additional ROI warnings omitted from console")

    if SAVE_QC_OVERLAY:
        save_overlay(rbpms_mean, masks, overlay_path)

    measurements = quantify_rois(
        image_path=path,
        roi_ids=range(1, roi_count + 1),
        get_mask=lambda i: masks == i,
        all_cells=masks > 0,
        projections=projections,
        marker_order=marker_order,
        z_count=z_count,
    )

    if SAVE_PER_IMAGE_CSV:
        measurements.to_csv(measurement_path, index=False)

    return {
        "measurements": measurements,
        "summary": {
            "image": path.name,
            "image_path": str(path),
            "axes": axes,
            "z_slices": int(z_count),
            "channel_count": int(channel_count),
            "marker_order": "|".join(marker_order),
            "rbpms_channel_1based": int(rbpms_channel + 1),
            "rbpms_roi_count": roi_count,
            "roi_zip_count": int(roi_zip_count) if np.isfinite(roi_zip_count) else np.nan,
            "mask_reused": bool(reused_mask),
            "roi_export_warning_count": len(roi_warnings),
            "status": "success",
        },
    }


# ============================================================
# MAIN
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="RBPMS segmentation (Cellpose cpdino) and per-cell marker measurement."
    )
    ap.add_argument("--input-dir", type=Path, default=Path("."),
                    help="Folder containing the multichannel TIFFs (searched recursively).")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Process only the first N source TIFFs (for testing).")
    ap.add_argument("--no-reuse-masks", action="store_true",
                    help="Re-run Cellpose even if a mask file already exists.")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="Run on CPU if CUDA is unavailable (very slow).")
    return ap.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()

    master_csv = input_dir / "RBPMS_all_cells_measurements.csv"
    summary_csv = input_dir / "RBPMS_image_summary.csv"
    log_csv = input_dir / "RBPMS_pipeline_log.csv"
    settings_json = input_dir / "RBPMS_pipeline_settings.json"

    print("\n" + "=" * 88)
    print("RBPMS PIPELINE - script 1")
    print("mean projection -> cpdino -> Fiji ROI ZIP -> marker measurements")
    print("=" * 88)
    print(f"Input directory:\n{input_dir}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    use_gpu = torch.cuda.is_available()
    print(f"\nPyTorch: {torch.__version__}   CUDA available: {use_gpu}")
    if use_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Use --allow-cpu to run on CPU anyway.")

    all_tiffs = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )
    source_files = find_source_tiffs(input_dir)
    skipped = [p for p in all_tiffs if is_generated_tiff(p)]
    if args.max_files is not None:
        source_files = source_files[: args.max_files]

    print(f"\nAll TIFFs found: {len(all_tiffs)}")
    print(f"Generated TIFFs skipped: {len(skipped)}")
    print(f"Source TIFFs to process: {len(source_files)}")

    if not source_files:
        print("No source TIFFs found.")
        return

    print(f"\nLoading {MODEL_NAME} once for the entire batch...")
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=MODEL_NAME)

    all_measurements, summaries, logs = [], [], []

    for i, path in enumerate(source_files, start=1):
        print(f"\n[{i}/{len(source_files)}] {path.name}")
        try:
            result = process_one(path, model, reuse_masks=not args.no_reuse_masks)
            all_measurements.append(result["measurements"])
            summaries.append(result["summary"])
            logs.append({"image": path.name, "status": "success", "error": ""})
        except Exception as exc:
            print("\n" + "!" * 88 + f"\nERROR\n{exc}\n" + "!" * 88)
            summaries.append({
                "image": path.name, "image_path": str(path),
                "status": "failed", "error": str(exc),
            })
            logs.append({
                "image": path.name, "status": "failed",
                "error": repr(exc), "traceback": traceback.format_exc(),
            })

        if use_gpu:
            torch.cuda.empty_cache()

    if all_measurements:
        master = pd.concat(all_measurements, ignore_index=True, sort=False)
        master.to_csv(master_csv, index=False)
        print(f"\nMaster cell table:\n{master_csv}")
        print(f"Total RBPMS cells measured: {len(master)}")

    pd.DataFrame(summaries).to_csv(summary_csv, index=False)
    pd.DataFrame(logs).to_csv(log_csv, index=False)

    settings = {
        "input_dir": str(input_dir),
        "known_markers": KNOWN_MARKERS,
        "rbpms_name": RBPMS_NAME,
        "model_name": MODEL_NAME,
        "flow_threshold": FLOW_THRESHOLD,
        "cellprob_threshold": CELLPROB_THRESHOLD,
        "min_size": MIN_SIZE,
        "normalize": NORMALIZE,
        "tile_size": TILE_SIZE,
        "batch_size": BATCH_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "resample": RESAMPLE,
        "background_gap_pixels": BACKGROUND_GAP_PIXELS,
        "background_outer_pixels": BACKGROUND_OUTER_PIXELS,
        "min_background_pixels": MIN_BACKGROUND_PIXELS,
        "reuse_existing_masks": not args.no_reuse_masks,
    }
    with open(settings_json, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    print("\n" + "=" * 88 + "\nFINISHED\n" + "=" * 88)
    print(f"Image summary:\n{summary_csv}")
    print(f"Pipeline log:\n{log_csv}")
    print(f"Settings:\n{settings_json}")


if __name__ == "__main__":
    main()

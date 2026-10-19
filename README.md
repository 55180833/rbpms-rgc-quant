# RBPMS RGC Pipeline

**From raw multichannel confocal stacks to marker-classified retinal ganglion cells, in two scripts.**

![Python](https://img.shields.io/badge/python-3.10-blue) ![License](https://img.shields.io/badge/license-MIT-green)

A reproducible workflow that segments RBPMS+ retinal ganglion cells (RGCs) with Cellpose,
lets you curate the ROIs in Fiji, and then classifies each cell for additional RGC markers
(e.g. SMI32, FOXP2, CART, OPN4, SPP1, VGLUT2, TBR2) using local-background-corrected intensity.
It outputs per-cell tables, per-image summaries, and Fiji-ready ROI sets for every
single, double, and triple positive population.

## Why this exists
Manual counting of RGC subtypes across many images is slow and hard to reproduce.
This pipeline automates segmentation and measurement while keeping a human in the loop
for the two decisions that matter: which ROIs are real cells, and what counts as "positive."

## Workflow

```
Multichannel Z-stack TIFF
        │  (channel order read from filename)
        ▼
 Script 1: 01_rbpms_segmentation.py
   Z mean projection → Cellpose (cpdino) on the RBPMS channel
   → Fiji ROI ZIP + masks + QC overlay + per-cell measurements
        │
        ▼
 Script 2: 02_marker_classification.py
   Opens Fiji → you delete false ROIs → you pick 5 clear positives per marker
   → re-measures your edited ROIs → threshold per marker/image
   → single / double / triple populations
   → classified CSVs + ROI ZIPs for each population
```

## Repository contents

| File | Purpose |
|---|---|
| `common.py` | Shared marker list, filename parsing, TIFF handling, background ring and quantification |
| `01_rbpms_segmentation.py` | Segmentation and per-cell measurement |
| `02_marker_classification.py` | Fiji curation and marker classification |
| `requirements.txt` | Python dependencies |

**Keep all three `.py` files in the same folder.** Both scripts import `common.py`.

## Features
- Batch processing of whole folders (recursive)
- Marker/channel assignment parsed automatically from filenames
- GPU-accelerated Cellpose (cpdino), loaded once per batch
- Fiji ROI ZIP export (`Cell_00001` naming keeps ROIs and CSV rows in sync)
- Local background ring per cell (with global fallback) and corrected metrics
  (mean, median, p90, p95, signal-to-background, z-score)
- One persistent Fiji session during review (no repeated restarts)
- Expert-anchored thresholds: minimum of 5 user-confirmed positives
- "At least" and "exact" double/triple co-expression counts, with exportable ROI sets
- Saved calibration JSON for reproducibility and re-runs

## Installation
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
conda create -n rbpms python=3.10
conda activate rbpms
pip install -r requirements.txt
```
Requires an NVIDIA GPU with CUDA for Script 1 (CPU is possible with `--allow-cpu`, but very slow)
and [Fiji](https://fiji.sc/) for Script 2.

## Input naming convention
Channel order is read from the filename, left to right:
```
Retina1_rbpms_smi32_foxp2_cart.czi - ....tif
```
Every channel must have a recognized marker name, and the number of markers in the
filename must equal the number of TIFF channels. Recognized markers are listed in
`KNOWN_MARKERS` in `common.py`:

`rbpms, cart, foxp2, smi32, spp1, vglut2, opn4, opn4ai9, tbr2, sensor, cal`

To add a new marker, add its name to `KNOWN_MARKERS` in `common.py`.

## Usage
1. Run segmentation (add `--max-files 1` to test on one image first):
```bash
   python 01_rbpms_segmentation.py --input-dir "path/to/images"
```
2. Run classification (opens Fiji for curation):
```bash
   python 02_marker_classification.py --input-dir "path/to/images"
```
3. In Fiji, delete false ROIs (do **not** rename ROIs). Then, in the terminal, enter
   5 clearly positive ROI IDs for each marker when prompted (or type `skip`).

### Command-line options
| Script | Option | Meaning |
|---|---|---|
| 1 | `--input-dir` | Folder with the TIFFs (searched recursively) |
| 1 | `--max-files N` | Process only the first N images |
| 1 | `--no-reuse-masks` | Re-run Cellpose even if a mask already exists |
| 1 | `--allow-cpu` | Run without a GPU |
| 2 | `--input-dir` | Folder with the TIFFs and `*_RBPMS_ROIs.zip` files |
| 2 | `--fiji-exe` | Full path to the Fiji executable (auto-detected if omitted) |
| 2 | `--recalibrate` | Ignore saved anchors and ask again |

## Outputs
| File | Content |
|---|---|
| `*_RBPMS_ROIs.zip` | Fiji ROIs, one per cell |
| `*_RBPMS_masks.tif` | Cellpose label mask |
| `*_RBPMS_overlay.png` | QC overlay |
| `RBPMS_all_cells_measurements.csv` | All cells, all markers (Script 1) |
| `RBPMS_curated_measurements.csv` | Re-measured cells after your Fiji edits (Script 2) |
| `RBPMS_all_cells_classified.csv` | Per-cell positive/negative calls |
| `RBPMS_simple_report.csv` | % of RBPMS+ cells per population |
| `RBPMS_single_marker_summary.csv` | Single-marker counts |
| `RBPMS_double_triple_summary.csv` | Co-expression counts |
| `RBPMS_roi_curation_summary.csv` | ROIs before/after your review |
| `RBPMS_classified_ROIs/` | ROI ZIPs per positive population |
| `RBPMS_positive_anchor_calibration.json` | Saved anchors and thresholds |

## Method notes (please read)
- Intensities are measured on **raw mean-Z projections**, never on contrast-stretched copies.
- Classification metric: `corrected_p95` (ROI 95th percentile minus local background median).
- Threshold = the lowest score among the 5 positives you select, so it reflects
  your judgement. Choose anchors that include weak-but-real cells.
- Thresholds are calibrated **per image and per marker**.
- Background is a ring around each cell (3 px gap, 12 px outer), excluding other cells.

## Limitations
- Thresholds depend on user-chosen anchors; describe your selection criteria in publications.
- Segmentation was validated on [tissue type, e.g. mouse retinal whole-mount, magnification] images.
- Tested on Windows with [GPU model]; Linux/macOS Fiji paths are untested.
- Fiji ROIs must stay polygon/freehand and keep their `Cell_XXXXX` names.

## Citation
If you use this pipeline, please cite this repository and:
- Stringer, C., Wang, T., Michaelos, M., Pachitariu, M. *Cellpose: a generalist algorithm for cellular segmentation.* Nat Methods (2021).
- Schindelin, J. et al. *Fiji: an open-source platform for biological-image analysis.* Nat Methods (2012).

## License
MIT License, see `LICENSE`.

## Contact
Saman Behbodi Tanourlouee. Behboodisaman09@gmail.com

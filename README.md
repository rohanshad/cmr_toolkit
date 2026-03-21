# Cardiac MRI Toolkit

[![Nature Biomedical Engineering](https://img.shields.io/badge/Nature_Biomedical_Engineering-10.1038%2Fs41551--026--01637--3-016795)](https://www.nature.com/articles/s41551-026-01637-3)

A preprocessing pipeline for cardiac MRI DICOM studies, converting raw acquisition files into structured, ML-ready HDF5 datasets. Supplementary repository for work performed in the paper: [A Generalizable Deep Learning System for Cardiac MRI](https://www.nature.com/articles/s41551-026-01637-3)

![summary_usage](https://github.com/rohanshad/cmr_toolkit/blob/5b6055dc059aeccb50bd78d106be4b88eccabe31/media/summary_usage.png)

---

## Overview

`cmr_toolkit` handles the full preprocessing lifecycle for multi-institutional cardiac MRI data — from raw tar.gz DICOM archives through standardized array storage — with built-in data integrity validation and cloud integration. All scripts scale linearly to ~64 CPU cores and can process upwards of 100k MRI scans in under 3 hours.

---

## Pipeline Architecture

```
DICOM Archives (tar.gz)
        ↓
[preprocess_mri.py]     ← parallel DICOM to HDF5 conversion, multi-institution
        ↓
HDF5 Filestore          ← per-patient, per-accession, arrays shaped [frames, H, W] or [frames, channels, H, W] (if RGB)
        ↓
[generate_checksums.py] ← SHA256 validation against ground-truth manifest

```

---

## HDF5 Output Format

After `preprocess_mri.py`, output is organized as `institution_anon_mrn/anon_accession.h5`. Each HDF5 file contains one dataset per MRI series, using the raw DICOM `SeriesDescription` string as the key:

```
upenn_Zx3da3244/
└── Gf3lv2173.h5
    ├── 4CH_FIESTA_BH           [frames, channels, H, W]   attrs: total_images, slice_frames
    ├── SAX_FIESTA_BH_1         [frames, channels, H, W]   attrs: total_images, slice_frames
    ├── SAX_FIESTA_BH_2         [frames, channels, H, W]   attrs: total_images, slice_frames
    └── STACK_LV3CH_FIESTA_BH   [frames, channels, H, W]   attrs: total_images, slice_frames
```

---

## Core Scripts

### `utils/tar_compress.py`
DICOM files are delivered in a variety of institutional patterns re: nested folder structures, filenaming conventions, and series folder distributions. This reads DICOM directories and writes compressed .tar.gz files in a standardized format, anonymizing MRN and Accession amongst other PHI if needed. Files are named: anon_mrn-anon_accession.tgz

### `utils/preprocess_mri.py`
Main entry point. Reads tar.gz DICOM archives, extracts pixel arrays, and writes compressed HDF5 files. Key behaviors:
- Handles institution-specific DICOM quirks (Stanford, UCSF, MedStar, UK Biobank, UPenn)
- Sorts frames by `SliceLocation` + `InstanceNumber` for correct temporal ordering
- Resizes frames via torchvision transforms (default 480px)
- Supports RGB and greyscale storage modes; greyscale reduces storage ~50–70%
- Default behaviour to downsample source float16 to uint8
- Optional direct upload to Google Cloud Storage during processing

```bash
python utils/preprocess_mri.py \
  -r /path/to/dicoms \
  -o /path/to/output \
  -i stanford \
  -c 16 \
  --channels rgb
```

| Argument | Description |
|---|---|
| `-r` / `--root_dir` | DICOM archive directory or GCS bucket (`gs://...`) |
| `-o` / `--output_dir` | HDF5 output location (required) |
| `-i` / `--institution` | Institution prefix: `stanford`, `ucsf`, `medstar`, `ukbiobank`, `upenn` |
| `-c` / `--cpus` | CPU cores for multiprocessing (default: 4) |
| `-s` / `--framesize` | Resize frames to this pixel size (default: 480) |
| `-z` / `--compression` | `gzip` or `lzf` (default: gzip) |
| `--channels` | `rgb` (default) or `grey` |
| `--gcs_bucket_upload` | Optional GCS bucket for direct upload |
| `-d` / `--debug` | Report statistics without converting |

### `utils/build_dataset.py`
Post-processes raw HDF5 output by renaming datasets from raw DICOM `SeriesDescription` strings to standardized view labels (`4CH`, `SAX`, `3CH`, `LAX`) using the lookup table in `series_descriptions_master.csv`. DEPRECATED

### `utils/generate_checksums.py`
Computes SHA256 checksums over HDF5 pixel data (not file headers) for reproducibility validation. Supports comparison against a reference manifest CSV to detect regressions between runs.

### `utils/dicom_metadata.py`
Scans DICOM archives to extract metadata (SeriesDescription, SliceLocation, Manufacturer, field strength, MRN, AccessionNumber) and outputs a CSV. 

### `utils/tar_compressor.py`
Compresses extracted DICOM folders back to tar.gz. Supports anonymization via a CSV crosswalk that remaps `(mrn, accession)` → `(anon_mrn, anon_accession)` during recompression.

### `utils/video_from_h5.py`
Converts HDF5 cine arrays to MP4 videos via FFmpeg for visual QC. Supports both greyscale and RGB modes.

### `utils/gcputils.py`
Google Cloud Storage utilities: asynchronous upload queue, GCS bucket mount/unmount via `gcsfuse`, and disk-full throttling (pauses pipeline when temp storage exceeds 90%).

### `utils/ukb_downloader.py`
Wrapper around the `ukbfetch` CLI for bulk UK Biobank downloads. Chunks large bulk files into 1000-row batches and runs parallel downloads (default: 20 concurrent connections).

### `utils/dicom_deid_mri.py` / `utils/llm_deid.py`
De-identification pipeline. `llm_deid.py` dispatches local Ollama LLM instances across multiple GPU devices to detect PHI in free-text DICOM fields and clinical reports.

---

## Configuration

### `local_config.yaml`
Device/cluster-specific settings (paths, CPU counts, GCS bucket name, Slack credentials). This file is machine-specific and not committed. Run `docker_prep.py` to generate a `.env` from it before running the Docker pipeline. Example structure:

```yaml
global_settings:
  bucket_name: 'your_gcs_bucket'
  slack_bot_token: 'xoxb-...'

sherlock:           # per-machine block
  tmp_dir: '/scratch/tmp'
  num_cpus: 48
```

### `series_descriptions_master.csv`
Lookup table mapping raw `SeriesDescription` DICOM strings → standardized view names (`4CH`, `SAX`, `3CH`, etc.). The `counts` column reflects occurrence frequency and helps verify coverage. If your institution's series descriptions are missing, run `dicom_metadata.py` on your data first and add entries manually. DEPRECATED

---

## Docker Pipeline & Pre-Push Validation

The full preprocessing pipeline runs inside a reproducible Docker environment (Ubuntu 24.04, Python 3.13). `docker-compose.yml` orchestrates sequential preprocessing across all supported institutions in both RGB and greyscale modes, followed by checksum generation and comparison against the ground-truth manifest.

A pre-push git hook enforces this automatically on every `git push`:

```
git push
  → docker_prep.py         (regenerate .env from local_config.yaml)
  → docker compose up      (build image, run full multi-institution pipeline)
  → generate_checksums.py  (compare output against checksum manifest)
  → push proceeds only if all checksums match
```

This blocks any commit that breaks a known-working preprocessing result from reaching the remote.

---

## Supported Institutions

| Institution | Prefix | Notes |
|---|---|---|
| Stanford | `stanford` | Standard DICOM metadata |
| UCSF | `ucsf` | Standard DICOM metadata |
| MedStar | `medstar` | MRN/accession extracted from filename |
| UK Biobank | `ukbiobank` | SAX split across multiple folders; EID-based naming |
| UPenn | `upenn` | MRN/accession extracted from filename |

---

## Machine Learning Integration (`engine/`) - TBD

- **`engine/torch_dataset.py`**: PyTorch `Dataset` class that reads HDF5 files directly, supports train/val/test CSV-driven splits, random or full-frame sampling, and optional transforms.
- **`engine/labeller.py`**: PyTorch Lightning module for CMR view/modality classification using a Facebook DINO ViT-S/16 backbone. In active development.

---

## Citation
If you use this codebase, or otherwise found our work valuable, please cite:
```
@article{shad2026generalizabledeeplearningcardiac,
      title={A Generalizable Deep Learning System for Cardiac MRI},
      author={Rohan Shad and Cyril Zakka and Dhamanpreet Kaur and Robyn Fong and Ross Warren Filice and John Mongan and Kimberly Kalianos and Nishith Khandwala and David Eng and Matthew Leipzig and Walter Witschey and Alejandro de Feria and Victor Ferrari and Euan Ashley and Michael A. Acker and Curtis Langlotz and William Hiesinger},
      journal={Nature Biomedical Engineering},
      year={2026},
      doi={10.1038/s41551-026-01637-3},
      url={https://www.nature.com/articles/s41551-026-01637-3},
}
```

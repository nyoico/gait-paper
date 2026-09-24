# From Force Plates to the Edge: A Compact Transformer for Gait Classification with Temporal Attribution and 3D Visualization

## Project Overview

This project classifies bilateral gait time series using a Transformer teacher, a compact student, and quantization-aware knowledge distillation (QKD). It includes model training, ONNX export, Raspberry Pi inference, and TIMING attribution for a 12-class classification task with five coarse groups.

## Conda Environment

Create and activate a Python 3.10 environment:

```bash
conda create -n gait-qkd python=3.10 pip -y
conda activate gait-qkd
```

Run the following installation commands from the repository root.

## Install Requirements

For training, evaluation, and visualization on a PC:

```bash
python -m pip install -r requirements_pc.txt
```

GPU execution requires a PyTorch installation compatible with the local CUDA environment.

For inference and optional serial acquisition on a Raspberry Pi:

```bash
python -m pip install -r requirements_rpi.txt
```

## Directory Structure

```text
gait-qkd-paper/
├── deploy/
│   └── labels.json                 # Class index-to-label mapping
├── figures/                        # Confusion matrices and learning curves
│   └── timing/                     # TIMING summary figures
├── results/
│   └── reference/
│       └── output/                 # Archived metrics, histories, and metadata
├── model.py                        # Transformer teacher
├── model_teacher_original.py       # Teacher implementation used by QKD
├── qkd_model.py                     # Student model and learned fake quantizers
├── qkd_common.py                    # Data preparation and evaluation utilities
├── train.py                        # Teacher training
├── train_student_fp.py              # Floating-point student training
├── train_qkd.py                     # Three-stage QKD training
├── inference.py                    # Teacher checkpoint evaluation
├── export_qkd_to_onnx.py            # FP32 ONNX export
├── export_qkd_qat_qdq.py            # ONNX export preserving learned QAT scales
├── quantize_onnx_int8.py             # Calibration-based INT8 quantization
├── check.py                        # Q/DQ ONNX graph validation
├── evaluate_onnx.py                 # ONNX accuracy evaluation
├── run_onnx_rpi.py                  # ONNX inference and benchmarking
├── export_normalization.py         # Training-data normalization statistics
├── pi_receive.py                   # ESP32 serial packet reception
├── prepare_gait_nne.py              # Teacher export for Unreal NNE
├── timing.py                       # TIMING attribution implementation
├── explain_timing.py               # Attribution computation
├── plot_timing_class_heatmaps.py    # Class-level attribution heatmaps
├── plot_figures.py                  # Confusion matrices and learning curves
├── plot_style.py                    # Shared figure styling
├── requirements_pc.txt             # PC dependencies
├── requirements_rpi.txt            # Raspberry Pi dependencies
├── FILE_MANIFEST.csv               # File provenance at repository assembly
└── README.md
```

Local datasets belong in `processed_data/`. Training generates checkpoints in `checkpoints/` and results in `output/`; exported models and their metadata are written to `deploy/`. Datasets, trained checkpoints, and exported models are not included in the repository.

# Gait Classification with Quantization-Aware Knowledge Distillation

## Overview

This repository provides a research implementation for classifying bilateral gait time series using a Transformer teacher, a compact floating-point student, and quantization-aware knowledge distillation (QKD). The workflow comprises supervised model training, three-stage QKD, ONNX export, deployment evaluation on PCs and Raspberry Pi devices, and temporal attribution with TIMING.

The classification task comprises 12 fine-grained classes and five coarse groups. The repository includes source code, selected figures, and archived experimental summaries. Raw data, processed datasets, trained checkpoints, and exported models must be obtained or generated separately.

## Methodology

### Model architecture

The teacher embeds the left and right sensor sequences separately using pointwise convolutions and batch normalization, adds the two embeddings, and applies sinusoidal positional encoding. A Transformer encoder-decoder produces class logits from a single decoder input token. The student retains this structure with a reduced embedding dimension and feed-forward width and introduces learnable fake quantizers.

| Configuration | Teacher | Student |
| --- | ---: | ---: |
| Input channels per side | 5 | 5 |
| Embedding dimension | 256 | 192 |
| Attention heads | 8 | 6 |
| Encoder / decoder layers | 4 / 4 | 4 / 4 |
| Feed-forward dimension | 1,024 | 768 |
| Dropout | 0.1 | 0.1 |
| Output classes | 12 | 12 |

The floating-point (FP32) baseline trains the student with quantization disabled. QKD initializes this student from its FP32 checkpoint and enables learned step-size fake quantization, with eight-bit weights and activations (W8A8) by default. Fake quantization simulates quantization during floating-point computation; integer execution is addressed separately during deployment.

### Distillation procedure

The QKD schedule in [train_qkd.py](train_qkd.py) consists of three consecutive stages:

| Stage | Training procedure | Default epochs |
| --- | --- | ---: |
| Self-studying (SS) | Optimize the quantized student using supervised cross-entropy. | 30 |
| Co-studying (CS) | Update both teacher and student using supervised and mutual distillation objectives. | 30 |
| Tutoring (TU) | Fix the teacher and refine the student using supervised learning and distillation. | 40 |

During CS and TU, the student objective is

$$
\mathcal{L}_{S}
=
\operatorname{CE}(y, z_S)
+
\tau^2 \operatorname{KL}
\left(
\operatorname{softmax}(z_T/\tau)
\;\middle\|\;
\operatorname{softmax}(z_S/\tau)
\right),
$$

where $z_T$ and $z_S$ are the teacher and student logits, respectively, and $\tau=2$ is the default temperature. The teacher distribution is detached when computing the student objective. During CS, the teacher uses the analogous objective with the roles reversed and a detached student distribution. SS uses cross-entropy alone.

### Data representation and evaluation protocol

Each sample contains synchronized left and right inputs with shape `[5, T]`. The loaders accept arrays shaped `[N, 5, T]` or `[N, T, 5]`, where `N` denotes the sample count. The reference configuration uses `T=101`; training sets the model sequence length from the supplied data. Channel order must match the original preprocessing convention. The repository does not specify the physical interpretation or units of the channels.

The class index order is:

```text
HC, H_P, H_C, H_F, K_P, K_F, K_R, A_F, A_R, A_L, C_F, C_A
```

Coarse labels follow the order `HC, H, K, A, C`. Five-class accuracy is computed by mapping the fine-class argmax and the ground-truth label to their respective groups.

The default scripts use seed 42 and reserve approximately 3% of the supplied training samples for validation. Normalization statistics are computed independently for each side and channel over the full supplied training arrays, **before** the validation split. Consequently, validation samples contribute to these statistics. The QKD stages reuse the split indices and normalization statistics stored in the FP32 student checkpoint.

This is a sample-level validation procedure. The repository does not establish subject- or session-independent separation, and it does not construct the externally supplied train/test split. Checkpoint selection uses 12-class validation accuracy.

## Repository organization

| File or directory | Purpose |
| --- | --- |
| `model.py`, `model_teacher_original.py` | Teacher implementations used by training, distillation, and inference scripts. |
| `qkd_model.py`, `qkd_common.py` | Student architecture, learned fake quantization, data preparation, and evaluation utilities. |
| `train.py`, `train_student_fp.py`, `train_qkd.py` | Teacher, FP32 student, and QKD training. |
| `inference.py` | Teacher checkpoint evaluation. |
| `export_qkd_to_onnx.py`, `quantize_onnx_int8.py` | FP32 export and calibration-based static INT8 quantization. |
| `export_qkd_qat_qdq.py`, `check.py` | Export of learned QAT scales as Q/DQ nodes and graph inspection. |
| `evaluate_onnx.py`, `run_onnx_rpi.py` | ONNX accuracy evaluation and inference benchmarking. |
| `timing.py`, `explain_timing.py`, `plot_timing_class_heatmaps.py` | TIMING attribution and class-level visualizations. |
| `plot_style.py`, `plot_figures.py` | Shared figure styling, confusion matrices, and learning curves. |
| `pi_receive.py`, `export_normalization.py`, `prepare_gait_nne.py` | Optional serial acquisition, normalization export, and Unreal NNE export. |
| `deploy/labels.json` | Fine-class index-to-label mapping. |
| `results/reference/output/` | Archived summaries, training histories, classification reports, and attribution metadata. |
| `figures/` | Selected confusion matrices, learning curves, and TIMING summary figures. |
| `FILE_MANIFEST.csv` | File provenance and SHA-256 hashes recorded when the repository was assembled. |

New runs write to `checkpoints/`, `output/`, and `deploy/`. Archived records are retained separately in `results/reference/`.

## Experimental setup

### Environment

Use Python 3.10 or later and execute commands from the repository root.

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on Linux/macOS, then install the PC dependencies:

```bash
python -m pip install -r requirements_pc.txt
```

GPU execution requires a PyTorch installation compatible with the local CUDA environment. The requirements files are dependency lists rather than complete environment lock files. `matplotlib>=3.10` is required for the default plotting colormaps. Raspberry Pi dependencies, including `pyserial` for optional serial acquisition, are listed in `requirements_rpi.txt`.

### Required data

Supply the following eight NumPy files:

```text
processed_data/
├── X_left_train.npy
├── X_right_train.npy
├── X_left_test.npy
├── X_right_test.npy
├── y_left_train.npy
├── y_right_train.npy
├── y_left_test.npy
└── y_right_test.npy
```

Label arrays contain one class string per sample. Left and right arrays must have matching sample counts, ordering, and labels. A complete pipeline from raw measurements to these arrays is not included; reproduction therefore begins with externally prepared data.

### Training and teacher evaluation

Run the training stages in order:

```bash
python train.py
python train_student_fp.py
python train_qkd.py
```

These scripts use configuration constants near the top of each file rather than command-line arguments. Teacher and FP32 student training default to a batch size of 32, an initial learning rate of `3e-4`, weight decay of `1e-4`, label smoothing of 0.05, and up to 300 epochs with an early-stopping patience of 20 epochs.

| Training stage | Principal checkpoint(s) | Results directory |
| --- | --- | --- |
| Teacher | `checkpoints/best_model.pt` | `output/` |
| FP32 student | `checkpoints/best_student_fp.pt` | `output/student_fp/` |
| QKD | `checkpoints/best_student_ss.pt`, `checkpoints/best_qkd_cs.pt`, `checkpoints/best_student_qkd.pt` | `output/qkd/` |

QKD requires both the teacher and FP32 student checkpoints. Evaluate the standard teacher checkpoint with:

```bash
python inference.py --checkpoint checkpoints/best_model.pt --data-dir processed_data
```

The documented workflow uses the standard model architecture. Legacy architecture-v2 checkpoints from excluded Optuna experiments are outside this workflow.

## ONNX export and deployment evaluation

### FP32 export and calibration-based INT8 quantization

```bash
python export_qkd_to_onnx.py --checkpoint checkpoints/best_student_qkd.pt --output deploy/student_qkd_fp32.onnx
python quantize_onnx_int8.py --input deploy/student_qkd_fp32.onnx --output deploy/student_qkd_int8.onnx --processed-dir processed_data --normalization deploy/normalization.npz --calibration-samples 512
```

The exporter writes an FP32 graph with fake quantization disabled, together with `normalization.npz` and `labels.json` in the output directory. Static quantization then estimates new scales from the supplied training arrays; the scales learned during QKD are not retained by this route. The default calibration method is min-max. Alternative settings include `--calibration-method percentile --calibration-samples 1024`.

### Export retaining learned QAT scales

```bash
python export_qkd_qat_qdq.py --checkpoint checkpoints/best_student_qkd.pt --output deploy/student_qkd_qat_qdq.onnx
python check.py
```

This route represents the learned quantization scales with `QuantizeLinear`/`DequantizeLinear` (Q/DQ) nodes and requires no separate calibration. `check.py` validates the graph at the default output path and checks for these nodes. Their presence alone does not establish integer-kernel coverage or a deployment speedup; execution depends on the runtime backend.

### Accuracy and runtime measurement

The same command-line interface supports PC and Raspberry Pi evaluation:

```bash
python run_onnx_rpi.py --model deploy/student_qkd_fp32.onnx --output output/fp32_results.json --predictions output/fp32_predictions.csv
python run_onnx_rpi.py --model deploy/student_qkd_int8.onnx --output output/int8_results.json --predictions output/int8_predictions.csv
python run_onnx_rpi.py --model deploy/student_qkd_qat_qdq.onnx --output output/qat_qdq_results.json --predictions output/qat_qdq_predictions.csv
```

With the repository layout, the default inputs are located in `processed_data/` and metadata in `deploy/`. Accuracy is evaluated when `y_left_test.npy` is available; otherwise, the script records predictions without target-based metrics. Use `--index 0` for a single sample and `--threads` to set the inference thread count.

Alternatively, edit `MODEL_PATH`, `NORMALIZATION_PATH`, `LABELS_PATH`, and `OUTPUT_PATH` in `evaluate_onnx.py` and run:

```bash
python evaluate_onnx.py
```

This script does not accept command-line options. Runtime measurements primarily cover model execution rather than complete data loading and output serialization. Report the device, operating system, Python and ONNX Runtime versions, thread count, and power configuration alongside timing results. RSS measurements describe process memory usage.

### Raspberry Pi execution

Place the following files in a single directory on the Raspberry Pi:

```text
run_onnx_rpi.py
requirements_rpi.txt
student_qkd_int8.onnx
normalization.npz
labels.json
X_left_test.npy
X_right_test.npy
y_left_test.npy          # Required only for accuracy evaluation
```

Then run:

```bash
python -m pip install -r requirements_rpi.txt
python run_onnx_rpi.py --threads 2
```

The default outputs are `onnx_dataset_predictions.csv` and `onnx_dataset_results.json`, containing dataset predictions, evaluation results when labels are available, and a batch-one benchmark.

## Temporal attribution and visualization

The TIMING implementation applies segment-masked integrated gradients to the teacher model. Run a small attribution experiment and aggregate the outputs by class:

```bash
python explain_timing.py --max-samples 32 --output-dir output/timing_example
python plot_timing_class_heatmaps.py --input-dir output/timing_example
```

Use `--max-samples 0` for the complete test set. The default target is the ground-truth class; `--target-mode predicted` selects the predicted class. The reference baseline is zero in the normalized input space.

The option `--class-level 5` explains coarse-group probabilities obtained by summing the corresponding fine-class probabilities. This attribution target differs from the fine-argmax mapping used to calculate five-class accuracy. Attributions characterize model responses to the input and do not establish causal relationships.

The default color schemes are `default`, `cividis`, and `managua`. Select a scheme with, for example, `--color-scheme cividis`. The plotting code saves color limits in accompanying metadata.

Generate confusion matrices and learning curves with:

```bash
python plot_figures.py
```

Confusion matrices require trained checkpoints and test data. Learning curves use `output/history.json`, `output/student_fp/history.json`, and `output/qkd/history.json`. To use archived histories, copy the desired files from `results/reference/output/` into the corresponding output locations after selecting the run to visualize; preserve any existing run outputs. Matching checkpoints and datasets must be supplied separately for confusion matrices.

Regenerating TIMING heatmaps requires the attribution arrays produced by `explain_timing.py`; the archived summary CSV and JSON files are insufficient. Color limits for the included TIMING figures are available in the [archived heatmap metadata](results/reference/output/timing_12cls_ground_truth/class_heatmaps_target/class_heatmap_metadata.json).

## Archived experimental results

The following values are transcribed from the included run summaries. They are historical observations and have not been reproduced or re-evaluated as part of this repository revision.

| Model | Reported parameters | 12-class test accuracy | 5-class test accuracy | Source |
| --- | ---: | ---: | ---: | --- |
| Teacher | 7,383,052 | 90.3210% | Not reported in summary | [Teacher summary](results/reference/output/summary.json) |
| FP32 student | 4,161,206 | 90.5717% | 91.6750% | [Student summary](results/reference/output/student_fp/summary.json) |
| QKD student, TU | 4,161,206 | 90.9729% | 92.0762% | [QKD summary](results/reference/output/qkd/summary.json) |

The archived test set contains 1,994 samples. Class `C_A` has zero test support, so the reported accuracy does not establish performance on all 12 classes. Class-specific results are provided in the [QKD classification report](results/reference/output/qkd/student_qkd_test_fine_classification_report.csv). These summaries do not establish statistical significance or performance variability across repeated runs.

![Archived QKD student confusion matrix for the 12-class task](figures/qkd_tu_confusion_12class_percent.png)

*Archived confusion matrix for the QKD student after tutoring. The figure is retained from an earlier experiment; correspondence to a specific checkpoint has not been independently verified.*

The archived PyTorch QKD timing measurements use fake-quantized floating-point execution and should not be interpreted as INT8 deployment benchmarks. Exported models require separate accuracy and runtime evaluation on the target device.

## Optional interfaces

- **ESP32 serial acquisition:** `pi_receive.py` receives sensor packets. Configure `SERIAL_PORT` and `USE_MODEL_INFERENCE` near the top of the file. Inference is disabled by default; enabling it requires model and normalization files alongside the script. ESP32 transmitter firmware is not included.
- **Normalization export:** `export_normalization.py` computes statistics from the training arrays. Standard QKD deployment should use the checkpoint-associated statistics written by the ONNX exporter.
- **Unreal NNE:** `python prepare_gait_nne.py --output ue_export` exports the teacher and performs numerical checks. This is separate from the QKD deployment workflow. Normalization is embedded in the exported graph, so inputs must not be normalized a second time. An Unreal project is not included.

## Reproducibility and artifact provenance

The released artifacts support inspection of the implementation and reproduction from compatible processed data. Full reproduction from raw measurements requires additional preprocessing code, data, and experimental metadata.

- Raw and processed data, subject/session identifiers, per-sample predictions and attributions, trained weights, ONNX models, normalization archives, and Optuna databases are excluded.
- Archived summaries and figures lack checkpoint hashes and complete environment records. Paths stored inside archived JSON files refer to the original run locations.
- Earlier ONNX evaluation summaries are excluded because inconsistent values could not be associated unambiguously with distinct runs.
- Legacy backups, duplicate deployment directories, and incompatible Optuna training code are excluded from the documented workflow.
- Dependency versions are not fully pinned. Reproduction records should retain the exact configuration, dataset split, environment, and model artifacts used for each experiment.

[FILE_MANIFEST.csv](FILE_MANIFEST.csv) records source paths, destination paths, file sizes, and SHA-256 hashes **at the time of repository assembly**. It is a provenance snapshot, not a checksum list for the current revision: subsequent English translations and documentation edits are not reflected in those hashes. The README, Git configuration files, and manifest itself were outside the original manifest scope.

The repository does not provide finalized publication metadata or a code license. No citation or licensing terms are inferred here, and dataset distribution terms must be established with the data provider.

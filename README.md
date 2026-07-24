# OA-Continuum: Anatomy-Aware Deep Learning Platform for Musculoskeletal Osteoarthritis Screening

> From infant hip dysplasia screening to adult knee osteoarthritis grading — in a single deployable radiograph-triage pipeline.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3%2B-purple.svg)](https://github.com/ultralytics/ultralytics)
[![Gradio](https://img.shields.io/badge/Gradio-4.44%2B-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Osteoarthritis (OA) is the leading musculoskeletal cause of disability worldwide. Automated screening today is fragmented: deep learning systems handle either **adult knee KL grading** or **infant hip dysplasia screening**, never both — despite strong clinical evidence that untreated infant hip dysplasia is a major precursor of early-onset adult hip OA.

This project unifies both ends of the OA continuum inside **one anatomy-aware pipeline**:

1. **Anatomy Router** — classifies an input radiograph as *knee* or *pelvis*, and abstains on inputs it doesn't recognize.
2. **Knee KL Classifier** — grades knee OA severity (Kellgren–Lawrence 0–4) with an ordinal-aware head.
3. **Hip Segmenter** — localizes bilateral hip regions on pediatric pelvic radiographs, as the first stage of an early hip-OA risk screen.

The system is framed explicitly as a **triage-assistance tool for radiologists** — not a diagnostic device — with abstention behavior treated as a core safety feature rather than an afterthought.

<p align="center">
  <em>Radiograph → Anatomy Router → [Knee Branch | Hip Branch | Abstain] → Structured JSON Report + Overlay</em>
</p>

---

## Key Results

| Component | Metric | Value |
|---|---|---|
| **Anatomy Router** | Validation accuracy | **100%** (120/120) |
| **Knee KL Classifier** | Quadratic-weighted kappa (QWK) | **0.809** |
| | Mean absolute error (grade units) | 0.486 |
| | Overall accuracy | 59.7% |
| **Hip Segmenter** | Box mAP@50 | **0.691** |
| | Mask mAP@50 | 0.680 |
| | Exactly 2 hips detected | 83.9% of test images |
| **End-to-End** | Abstention-aware accuracy | **79.4%** @ 83.8% coverage |
| **Inference Latency** | GPU / CPU (median) | 31.5 ms / 195.7 ms per image |

Full per-class breakdowns, confusion matrices, and training curves are in the [paper](docs/paper.pdf) (Sections 5–6).

---

## Architecture

```
                         ┌─────────────────────────┐
                         │   Input Radiograph       │
                         └────────────┬────────────┘
                                      │
                    Greyscale → Resize+Pad → ImageNet Normalize
                                      │
                         ┌────────────▼────────────┐
                         │   Anatomy Router          │
                         │   EfficientNet-B0         │
                         │   max-softmax τ = 0.80    │
                         └──────┬──────────────┬─────┘
                     conf < 0.80│              │conf ≥ 0.80
                         ┌──────▼──────┐   ┌────▼─────────────┐
                         │  Abstain    │   │  Route to        │
                         │  (reject)   │   │  Specialist       │
                         └─────────────┘   └────┬─────────┬───┘
                                          knee   │         │  hip_pelvis
                                    ┌────────────▼──┐   ┌──▼─────────────────┐
                                    │ Knee KL        │   │ Hip Segmenter       │
                                    │ EfficientNetV2-S│  │ YOLO11s-seg         │
                                    │ + CORAL head    │  │ CSP + FPN-PAN       │
                                    │ + Grad-CAM      │  │ + hip-count check   │
                                    └────────┬───────┘   └──────────┬──────────┘
                                             │                      │
                                             └──────────┬───────────┘
                                                         ▼
                                        ┌─────────────────────────────┐
                                        │  Unified JSON Report          │
                                        │  + Overlay Image               │
                                        └──────────┬──────────────────┘
                                                    ▼
                                        ┌─────────────────────────────┐
                                        │  Gradio Web App (offline)     │
                                        └─────────────────────────────┘
```

---

## Repository Structure

```
oa-continuum/
├── training/
│   ├── knee/                 # KL-grading classifier training pipeline
│   │   ├── train.py
│   │   ├── dataset.py
│   │   ├── coral_head.py
│   │   └── configs/
│   └── hip_yolo/              # Hip segmentation training pipeline
│       ├── train.py
│       ├── data.yaml
│       └── configs/
├── platform/                  # Unified inference pipeline + web app
│   ├── app/
│   │   ├── pipeline.py        # Shared preprocessing (greyscale, resize-pad, normalize)
│   │   ├── router.py          # Anatomy router inference
│   │   ├── knee_infer.py      # KL classifier inference + Grad-CAM
│   │   ├── hip_infer.py       # Hip segmenter inference + count-check
│   │   └── report.py          # Unified JSON/Markdown report dataclass
│   ├── weights/                # router.pt, knee.pt, hip.pt (not tracked in git — see below)
│   ├── main.py                 # Gradio + FastAPI entrypoint
│   └── requirements.txt        # Inference-time dependencies only
├── docs/
│   └── paper.pdf
├── verify_install.py           # End-to-end smoke test (loads 3 checkpoints, dummy forward pass)
├── package_for_transfer.sh     # Bundles weights + platform code into oa_deploy.zip
├── requirements.txt             # Full training + inference dependencies
└── README.md
```

---

## Datasets

| Pathway | Dataset | Size | Source |
|---|---|---|---|
| Knee KL grading | OAI-derived Knee OA Severity Grading | 8,260 radiographs | [Kaggle](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity) |
| Hip segmentation | AV-DDH | 16,205 pelvic radiographs | [Mendeley Data](https://doi.org/10.17632/4gvcb6gmh2.2) |
| Anatomy router | Balanced subset of both above | 600 train / 120 val | Derived (see `training/router/`) |

Both source datasets are publicly released under permissive research licenses and are de-identified at source. No additional human-subject data was collected for this work.

> **Note:** Raw datasets are not bundled in this repository due to size and licensing. See `training/*/README.md` for download and preprocessing instructions.

---

## Installation

### Training environment (GPU required)

```bash
git clone https://github.com/<your-username>/oa-continuum.git
cd oa-continuum
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** PyTorch ≥ 2.2, Ultralytics ≥ 8.3, timm ≥ 1.0.7, Albumentations, Gradio ≥ 4.44, FastAPI ≥ 0.110

### Inference-only environment (CPU, offline-capable)

```bash
cd platform
pip install -r requirements.txt
```

No internet access is required at inference time — checkpoints are loaded with `pretrained=False`, so no ImageNet weights are downloaded at runtime.

---

## Usage

### Train the knee KL classifier

```bash
python training/knee/train.py \
  --data-dir /path/to/knee-dataset \
  --epochs 40 \
  --lr 3e-4 \
  --batch-size 32
```

### Train the hip segmenter

```bash
python training/hip_yolo/train.py \
  --data training/hip_yolo/data.yaml \
  --epochs 100 \
  --imgsz 640 \
  --patience 20
```

### Run the deployed web app

```bash
cd platform
python main.py
```

Then open `http://localhost:7860` in your browser. Upload a radiograph and the app returns:
- Routed anatomy + router confidence
- KL grade + Grad-CAM overlay (knee) *or* hip masks + count (pelvis)
- Structured JSON report
- Manual anatomy-override option

### Verify a deployment package before shipping

```bash
python verify_install.py
```

Runs a dummy forward pass through all three checkpoints to confirm the offline package is functional before transferring to an air-gapped machine.

---

## Deployment Notes

- Package a transfer bundle with: `bash package_for_transfer.sh` → produces `oa_deploy.zip` containing `router.pt`, `knee.pt`, `hip.pt`, and the `platform/` code.
- Target machines only need a CPU and the dependencies in `platform/requirements.txt`.
- Delete Ultralytics `.cache` files before any cross-machine training — they contain absolute paths from the labeling machine and cause silent label-mismatch errors otherwise.
- Avoid naming any project folder `platform` if it shadows Python's standard library — rename for production use.

---

## Limitations

- **Single-source evaluation** — no external validation cohort yet; reported metrics likely overestimate cross-institution generalization.
- **No adult hip-OA grading** — the hip pathway is an early-risk *screen*, not a severity grader (see Future Work).
- **Imprecise hip masks at strict IoU** — mAP@50–95 drops sharply (0.295–0.342) vs. mAP@50, indicating polygon boundaries are approximate.
- **Single training seed** — no confidence intervals reported.
- **ImageNet-pretrained backbones** — domain-specific pretraining (RadImageNet, MedViT) is a likely improvement, untested here.

See Section 6.5 of the paper for full discussion.

## Future Work

- [ ] Adult hip-OA severity grading bridge (OAI hip subset / CHECK cohort / World COACH)
- [ ] External validation on non-OAI / non-AV-DDH cohorts
- [ ] Domain-pretrained backbones (RadImageNet, MedViT) ablation
- [ ] Calibrated abstention (temperature scaling, deep ensembles, evidential DL)
- [ ] Radiologist-in-the-loop evaluation study
- [ ] Federated training across institutions


---

## License

This project is released under the [MIT License](LICENSE). Datasets used retain their original publisher licenses — see the respective dataset pages for terms.

## Disclaimer

This is a **research prototype**, not a certified medical device. All outputs are intended for radiologist-assistive triage only and must not be used for standalone clinical diagnosis.

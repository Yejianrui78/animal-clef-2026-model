# AnimalCLEF 2026 — Improved Solution

> **Team:** SCNU-DT &nbsp;|&nbsp; **Account:** yejianrui &nbsp;|&nbsp; **Platform:** [Kaggle](https://www.kaggle.com/)  
> **Final Rank:** 44 / 230 (top 19.1%) &nbsp;|&nbsp; **Overall Score:** 0.33717

---

## 📊 Ranking Summary

| Dataset | ARI (Adjusted Rand Index) | Notes |
|---|---|---|
| SeaTurtleID2022 | **0.9150** | 🥇 Clustering performance outstanding |
| LynxID2025 | 0.2808 | IR/visible cross-modal challenge |
| SalamanderID2025 | 0.1620 | Hand-held capture, high viewpoint variance |
| TexasHornedLizards | — | Low inter-class variance |
| **Overall** | **0.33717** | **Rank 44 / 230** |

The four subsets present distinct challenges: SeaTurtleID2022 features standardized underwater images with clear individual markers; LynxID2025 mixes infrared (IR) and visible-light camera-trap images; SalamanderID2025 contains hand-held photos with extreme pose variation; TexasHornedLizards has very subtle inter-individual differences.

---

## 🧠 Method

### 1. Overall Pipeline

```
Raw Image
  │
  ├─ [Preprocessing] YOLO animal detection & crop (MegaDetector v1000)
  ├─ [Preprocessing] Dataset-specific pipelines (Lynx / Salamander)
  │
  ├─ [Feature Extraction] MegaDescriptor-L-384 (Swin-L, 384×384)
  ├─ [Feature Extraction] MiewID-msv3 (ViT, 512×512)
  │
  ├─ [Fusion] Weighted concatenation + L2 normalization
  ├─ [TTA] Multi-view test-time augmentation
  │
  └─ [Clustering] Agglomerative Clustering (average linkage)
       with n_clusters cap to prevent over-fragmentation
```

### 2. Feature Extraction — Dual-Model Fusion

We leverage two complementary pretrained models from the wildlife computer vision ecosystem:

- **MegaDescriptor-L-384** — Swin-Large backbone pretrained on diverse wildlife camera-trap datasets. Input size 384×384. Loaded via `timm` from `BVRA/MegaDescriptor-L-384`.
- **MiewID-msv3** — Vision Transformer trained on the MiewID individual-identification benchmark. Input size 512×512. Loaded via HuggingFace `transformers` from `conservationxlabs/miewid-msv3`.

**Fusion strategy:** Features from both models are concatenated with learnable per-dataset weights and L2-normalized:

$$\text{fused} = \text{L2-Norm}\left( [w_{\text{mega}} \cdot f_{\text{mega}} \;\|\; w_{\text{miew}} \cdot f_{\text{miew}}] \right)$$

where $w_{\text{mega}} + w_{\text{miew}} = 3.0$. Optimal weights are found by grid search on the training split (when available), maximizing ARI. For datasets without train labels, we use Silhouette score as an unsupervised proxy with prior transferred from SalamanderID2025.

### 3. Image Preprocessing

#### 3.1 YOLO Animal Detection & Cropping
- Uses **MegaDetector v1000** (`md_v1000.0.0-redwood.pt`) via YOLOv5 `torch.hub`
- Detects and crops the animal region with a 5% margin expansion
- Falls back to full image when no animal is detected
- Optional **SAM (Segment Anything Model)** refinement for tighter masks

#### 3.2 LynxID2025 — Cross-Modal Normalization
LynxID2025 contains both infrared (IR) and visible-light camera-trap images. To bridge the domain gap:

- **IR detection:** HSV saturation mean < 22.0 → classified as IR
- **Stage 2 (LAB alignment):** IR images have a/b channels set to neutral (128); color images undergo mild regression toward neutral. L channel is standardized to μ=128, σ=64 across all images.
- **Stage 3 (texture):** CLAHE contrast enhancement (clip=1.5 for IR, 2.0 for visible), followed by bilateral filtering (color) or Non-Local Means denoising (IR), and mild unsharp masking.

#### 3.3 SalamanderID2025 — Glare Removal & Texture Enhancement
Hand-held salamander photos suffer from flash glare, inconsistent lighting, and motion blur:

- **Glare removal:** High-luminance regions (L > 245) detected and inpainted via Telea algorithm
- **Luminance alignment:** L channel mapped to [40, 210]; A/B channels aligned to reference means
- **Texture enhancement:** Bilateral or NL-Means denoising + mild unsharp masking (amount=0.14)
- **Low-quality heuristic:** Images with extreme mean L or very low L std receive gentler processing

#### 3.4 Common Enhancements
- **Letterbox resize:** Aspect-ratio-preserving resize with edge-replication padding (avoiding black-border artifacts)
- **CLAHE lighting normalization** (optional, off by default)
- **Texture sharpening** via unsharp mask (optional, off by default)

### 4. Test-Time Augmentation (TTA)

Per-dataset multi-view TTA with mean-pooling:

| Dataset | Views |
|---|---|
| SalamanderID2025 | Original + H-Flip + V-Flip + 90° Rot + ColorJitter |
| SeaTurtleID2022 | Original + H-Flip + ColorJitter |
| LynxID2025 | Original + H-Flip |
| TexasHornedLizards | Original + H-Flip |

All views are individually L2-normalized before mean-pooling, followed by a final L2 normalization.

### 5. Clustering — Agglomerative with Cluster Cap

After extensive comparison on the training splits, **Agglomerative Clustering** (average linkage, cosine distance) consistently outperforms DBSCAN and HDBSCAN across all datasets.

**Key innovation — n_clusters cap:** The standard `distance_threshold` mode can produce excessive singleton clusters. We introduce an automatic upper bound:
When `distance_threshold` produces more clusters than the cap, the algorithm falls back to `n_clusters` mode with k = max_clusters, enforcing more aggressive merging.

**Parameter selection:**
- **With train split:** Grid search over `distance_threshold ∈ [0.06, 0.55]`, maximizing ARI against ground-truth identities. When fusion is enabled, the search is joint over both weight and threshold.
- **Without train split:** Per-dataset fallback thresholds (e.g., LynxID2025=0.35, SalamanderID2025=0.25) derived from prior experiments.

### 6. Fine-Tuning (Optional)

The code supports optional Triplet Loss fine-tuning via `wildlife_tools.BasicTrainer`:
- Identity-preserving train/val split (same individual never appears in both)
- Semi-hard triplet mining with margin=0.2
- Gradient accumulation for memory-constrained GPUs
- Checkpoints saved to `finetuned_models/`

Enabled via the `ENABLE_FINETUNING` flag. By default off — the pretrained features alone achieve competitive performance.

### 7. Feature Strategy Modes

The code supports five configurable feature strategies via `FEATURE_STRATEGY`:

| Strategy | Description |
|---|---|
| `fusion` | Mega‖Miew concatenation for all 4 subsets |
| `fusion_salamander_only` | Fusion for Salamander + TexasHornedLizards; SeaTurtle→Mega, Lynx→Miew |
| `per_dataset_baseline` | Per-dataset backbone selection (as baseline) |
| `global_mega` | MegaDescriptor only for all subsets |
| `global_miew` | MiewID only for all subsets |

---

## 📁 Repository Structure

```
.
├── improved_solution_v2.py   # Main solution script (2209 lines)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore                # Excludes data/models/third-party
└── assets/                   # Screenshots for verification
    ├── leaderboard.png       # Kaggle leaderboard ranking
    ├── account.png           # Account/profile page
    └── scores.png            # Per-dataset score breakdown
```

### External Dependencies (not included in this repo)

| Dependency | Source |
|---|---|
| MegaDetector v1000 weights | [GitHub Releases](https://github.com/agentmorris/MegaDetector/releases/tag/v1000.0) |
| YOLOv5 | `git clone https://github.com/ultralytics/yolov5` |
| MegaDescriptor-L-384 | HuggingFace: `BVRA/MegaDescriptor-L-384` |
| MiewID-msv3 | HuggingFace: `conservationxlabs/miewid-msv3` |
| AnimalCLEF 2026 Dataset | [Kaggle Competition](https://www.kaggle.com/c/animalclef-2026) |

---

## 🚀 Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/YOUR_USERNAME/animalclef-2026.git
cd animalclef-2026

# Install dependencies
pip install -r requirements.txt

# Download MegaDetector weights
# Place md_v1000.0.0-redwood.pt in the project root

# Clone YOLOv5 (for animal cropping)
git clone https://github.com/ultralytics/yolov5

# Download dataset from Kaggle into ./images/
```

### 2. Configure

Edit the experiment switches at the top of `improved_solution_v2.py`:

```python
FEATURE_STRATEGY = 'fusion'        # Recommended: dual-model fusion
USE_TTA = True                     # Multi-view test-time augmentation
USE_MEGADETECTOR_CROP = True       # YOLO animal detection & crop
GRID_SEARCH_ON_TRAIN = True        # Hyperparameter search on train split
```

### 3. Run

```bash
python improved_solution_v2.py
```

The script will:
1. Load the dataset (4 subsets from `images/`)
2. Load MegaDescriptor-L-384 and MiewID-msv3
3. Extract features with TTA
4. Search optimal clustering parameters on train splits
5. Generate `submission.csv`

### 4. Hardware Requirements

- **GPU:** NVIDIA GPU with ≥8 GB VRAM (tested on RTX 4060)
- **RAM:** ≥16 GB system memory
- **Disk:** ~30 GB for dataset images + model weights (~2 GB)

---

## 📈 Ablation & Design Decisions

Key findings from our experiments:

1. **Dual-model fusion > single model:** Concatenating Mega+Miew features consistently outperforms either backbone alone, especially for SalamanderID2025 (+0.03~0.05 ARI).

2. **Dataset-specific preprocessing matters:** The Lynx IR/visible alignment and Salamander glare removal each contribute measurable gains on their respective subsets.

3. **Agglomerative > DBSCAN > HDBSCAN:** On all four subsets, Agglomerative clustering with average linkage achieves the highest ARI. DBSCAN is competitive but sensitive to eps. HDBSCAN tends to over-fragment.

4. **n_clusters cap is critical:** Without the cap, Agglomerative with `distance_threshold` can produce up to 50% singleton clusters on some datasets. The cap reduces singletons without sacrificing ARI.

5. **Multi-view TTA helps rotation-variant datasets:** The 5-view TTA for SalamanderID2025 (including 90° rotation and vertical flip) is especially beneficial for hand-held photos with arbitrary orientations.

6. **Per-dataset weight search is worth it:** The optimal Mega/Miew fusion ratio differs substantially across datasets (Salamander ≈ 1.5:1.5, SeaTurtle ≠ 1:1), justifying per-dataset grid search.

---

## 🔍 Verification

See the `assets/` directory for official screenshots:

| Screenshot | Content |
|---|---|
| `assets/leaderboard.png` | Kaggle leaderboard showing rank **44/230** with score **0.33717** |
| `assets/account.png` | Kaggle account page showing username **yejianrui**, team **SCNU-DT** |
| `assets/scores.png` | Per-dataset score breakdown (ARI values) |

> **Note:** Please add these screenshots to the `assets/` folder before pushing to GitHub. The screenshots serve as verifiable evidence of the claimed ranking.

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@misc{animalclef2026-scnudt,
  author       = {SCNU-DT (yejianrui)},
  title        = {AnimalCLEF 2026 — Dual-Model Fusion with Dataset-Specific Preprocessing},
  year         = {2026},
  howpublished = {\url{https://github.com/YOUR_USERNAME/animalclef-2026}},
  note         = {Rank 44/230 (top 19.1\%), Overall Score 0.33717}
}
```

## 📄 License

MIT License

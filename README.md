# **TESS Exoplanet Transit Detection Pipeline (1D CNN)**

A reproducible pipeline for detecting transiting exoplanets in TESS light curves using a 1D convolutional neural network trained on confirmed exoplanet hosts, BATMAN‑simulated transits, and non‑planet control stars. The system identifies new candidates by combining CNN predictions with Box Least Squares (BLS) validation, enabling automated screening of large TESS sectors.

TESS produces millions of light curves per sector. Current algorithms like SPOC or QLP are optimized to find highly periodic, relatively bright, multiple-transit planet candidates orbiting solitary stars, essentially "ideal" transit cases. This pipeline extends the search into more challenging regimes, including faint targets, noisy photometry, and single‑transit events, enabling candidate detection in domains traditional algorithms often overlook.

---

## **1. Pipeline Architecture**

The pipeline operates in four sequential stages:

1. **Data Acquisition**
   Downloads confirmed transiting exoplanets from the NASA Exoplanet Archive (TAP).
   **Output:** `confirmed_exoplanets.csv`

2. **Dataset Construction**
   - **Real positives:** `generate_confirmed_dataset.py` → `confirmed_positive_vectors.npy`
   - **Synthetic positives:** `generate_synthetic_dataset.py` → `synthetic_positive_vectors.npy` (BATMAN simulations)
   - **Negative controls:** `generate_negative_dataset.py` → `negative_vectors.npy` (exoplanet-free TESS targets)

3. **Model Training**
   Trains a 1D CNN with dropout, batch normalization, and class-balanced loss on combined datasets.
   **Output:** `exoplanet_cnn_real.h5`

4. **Candidate Screening**
   Applies the CNN to new TESS sectors, filters known TICs, and validates top predictions with BLS periodogram analysis.
   **Outputs:** `bls_results.csv`, `undiscovered_candidate_TIC_*.png`

---

## **2. Environment Setup**

Activate the Python environment:

```bash
tf_env\Scripts\activate
```

### **Dependencies**
- `lightkurve` — TESS light curve retrieval and preprocessing
- `tensorflow` — 1D CNN training
- `astropy` — Box Least Squares (BLS) period search
- `pyvo` — NASA Exoplanet Archive TAP queries
- `batman-package` — synthetic transit light curve simulation
- `numpy`, `pandas`, `matplotlib` — data handling and visualization

Install BATMAN:

```bash
pip install batman-package
```

---

## **3. Usage Guide**

Execute the pipeline in sequential order:

### **Step 1 — Download Confirmed Exoplanets**
```bash
tf_env\Scripts\python.exe download_confirmed_exoplanets.py
```
Downloads confirmed transiting planets from NASA Exoplanet Archive.
**Output:** `confirmed_exoplanets.csv`

### **Step 2 — Generate Training Datasets**
```bash
tf_env\Scripts\python.exe generate_confirmed_dataset.py
tf_env\Scripts\python.exe generate_synthetic_dataset.py
tf_env\Scripts\python.exe generate_negative_dataset.py
```
Generates training vectors:
- `confirmed_positive_vectors.npy` — real confirmed exoplanet light curves
- `synthetic_positive_vectors.npy` — BATMAN-simulated transits
- `negative_vectors.npy` — exoplanet-free TESS targets

### **Step 3 — Train the CNN**
```bash
tf_env\Scripts\python.exe cnn_model.py
```
Trains a 1D CNN with class-weighted binary crossentropy.
**Output:** `exoplanet_cnn_real.h5`

### **Step 4 — Screen New TESS Sectors**
```bash
tf_env\Scripts\python.exe find_undiscovered_candidates.py
```
Applies model to Sector 21, filters known TICs, and runs BLS validation on CNN candidates.
**Outputs:**
- `bls_results.csv` — validated candidates with period, depth, SNR
- `undiscovered_candidate_TIC_*.png` — phase-folded diagnostic plots

---

## **4. Outputs**

| Component | Description |
|----------|-------------|
| `exoplanet_cnn_real.h5` | Trained 1D CNN model (binary classifier) |
| `bls_results.csv` | Validated candidates: TIC, period, depth, duration, BLS power |
| `undiscovered_candidate_TIC_*.png` | Phase-folded light curves of top BLS candidates |
| `confirmed_exoplanets.csv` | TAP-sourced confirmed transiting exoplanets |
| `confirmed_positive_vectors.npy` | Normalized light curves from confirmed hosts |
| `synthetic_positive_vectors.npy` | BATMAN-simulated transit light curves |
| `negative_vectors.npy` | Light curves from exoplanet-free TESS targets |

---

## **5. Methods / Technical Details**

- **Model Architecture**
  - 1D CNN with 3 convolutional blocks (16→32→64 filters), kernel sizes 7→5→3
  - Each block: Conv1D → BatchNorm → MaxPooling1D → Dropout(0.5)
  - Dense layers: 64-unit ReLU with L2 regularization and Dropout(0.5)
  - Output: Sigmoid for binary classification
  - Optimizer: Adam (lr=1e-4, weight_decay=1e-4)
  - Loss: Binary crossentropy with dynamic class weighting

- **Data Preprocessing**
  - Light curves: normalized to unit mean, clipped to [0.9, 1.1]
  - Interpolated to fixed length (2000 points) via linear interpolation
  - Excluded targets with noise σ > 0.02
  - Training data: 5,000 synthetic + 1,200 real positives + 1,500 negative controls

- **Training Strategy**
  - 50 epochs, batch size 64
  - Early stopping: saves best model by validation precision
  - Learning rate reduction: factor 0.5 on plateau (patience=5)

- **Validation Methodology**
  - Train/test split: 80/20, stratified by class
  - Performance metrics: precision, recall, accuracy
  - Training history logged to `training_history_real.png`

- **Assumptions**
  - Transits are periodic, symmetric, and detectable above noise
  - TIC IDs in NASA archive are accurate and complete
  - BLS can recover transits with depth ≥ 0.001 and duration ≤ 0.3 days

- **Limitations**
  - Model trained on Sector 21 data only; generalization to other sectors unverified
  - BATMAN simulations assume circular, zero-inclination orbits
  - Stellar variability modeled as red noise only — no flares or spots
  - No correction for stellar limb darkening beyond quadratic limb-darkening law
  - BLS validation uses fixed duration range (0.05–0.3 days)

---

## **6. Reproducibility**

- Random seed: `42` for train/test split in `cnn_model.py`
- Hardware: NVIDIA RTX 3090 (GPU-accelerated training)
- Software: Python 3.11, TensorFlow 2.12, BATMAN 2.4.8, astropy 5.3
- Training time: ~4 hours on GPU
- Results are deterministic with fixed seeds and identical environment
- All data generation and training scripts include explicit seeding

---

## **7. Project Structure**

```
TESS-project/
│── data/
│   └── toi-catalog_2026-06-23.csv
│── src/
│   ├── download_confirmed_exoplanets.py
│   ├── generate_confirmed_dataset.py
│   ├── generate_synthetic_dataset.py
│   ├── generate_negative_dataset.py
│   ├── cnn_model.py
│   ├── find_undiscovered_candidates.py
│   └── pipeline.py
│── models/
│   └── exoplanet_cnn_real.h5
│── results/
│   ├── bls_results.csv
│   └── undiscovered_candidate_TIC_*.png
│── training_history_real.png
│── README.md
```

This helps reviewers navigate your repo quickly.

---

## **8. Limitations & Future Work**

- **Known weaknesses**:
  - Model trained on Sector 21 only — performance on other sectors unverified
  - Stellar activity modeled as red noise only — no flares, spots, or pulsations
  - BLS validation uses fixed duration range (0.05–0.3 days)

- **Planned improvements**:
  - Train on multi-sector data to improve generalization
  - Incorporate stellar metadata (Teff, radius, log g) as auxiliary inputs
  - Add ensemble method comparing CNN with BLS-only detection

- **Possible extensions**:
  - Extend pipeline to K2 and JWST NIRSpec light curves
  - Integrate with TESS EPO and NASA Exoplanet Archive alert systems
  - Build public web interface for candidate submission and community review

- **Intentionally omitted**:
  - Transmission spectroscopy modeling
  - Multi-planet transit fitting
  - Real-time alerting system

---
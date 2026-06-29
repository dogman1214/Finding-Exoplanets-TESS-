
# **TESS Exoplanet Transit Detection Pipeline (1D CNN)**

A reproducible machine‑learning pipeline for identifying exoplanet transit signatures in *Transiting Exoplanet Survey Satellite* (TESS) light curves. The system integrates real confirmed exoplanet hosts, physics‑based synthetic transit models (BATMAN), and non‑planet control stars to train a 1D convolutional neural network (CNN) capable of scanning new TESS sectors for previously unreported candidates.

---

## **1. Pipeline Architecture**

The workflow is organized into four primary stages:

1. **Acquisition of Confirmed Exoplanet Hosts**  
   Retrieves all TESS‑observed confirmed planets from the NASA Exoplanet Archive (TTAP).  
   **Output:** `confirmed_exoplanets.csv`

2. **Dataset Construction**  
   - **Real positives:**  
     `generate_confirmed_dataset.py` → `confirmed_positive_vectors.npy`  
   - **Synthetic positives (BATMAN):**  
     `generate_synthetic_dataset.py` → `synthetic_positive_vectors.npy`  
   - **Negative controls:**  
     `generate_negative_dataset.py` → `negative_vectors.npy`

3. **Model Training**  
   Trains a 1D CNN on combined real, synthetic, and negative samples.  
   **Output:** `exoplanet_cnn_real.h5`

4. **Sector‑Level Candidate Search**  
   Applies the trained model to new TESS sectors, filters out known hosts/TOIs, and performs BLS validation on high‑confidence predictions.  
   **Outputs:** `bls_results.csv`, `folded_plots/`

---

## **2. Environment Setup**

Activate the project environment:

```bash
tf_env\Scripts\activate
```

### **Dependencies**
- `lightkurve` — TESS light curve access and preprocessing  
- `tensorflow` — 1D CNN model training  
- `astropy` — Box Least Squares (BLS) period search  
- `pyvo` — NASA Exoplanet Archive TAP queries  
- `batman-package` — analytic transit model generation  
- `numpy`, `pandas`, `matplotlib`

Install BATMAN:

```bash
pip install batman-package
```

---

## **3. Usage Guide**

### **Step 1 — Download Confirmed Exoplanets**
```bash
tf_env\Scripts\python.exe download_confirmed_exoplanets.py
```

### **Step 2 — Generate Training Datasets**
```bash
tf_env\Scripts\python.exe generate_confirmed_dataset.py
tf_env\Scripts\python.exe generate_synthetic_dataset.py
tf_env\Scripts\python.exe generate_negative_dataset.py
```

### **Step 3 — Train the CNN**
```bash
tf_env\Scripts\python.exe cnn_model.py
```

Produces:  
- `exoplanet_cnn_real.h5` (trained model)

### **Step 4 — Scan New TESS Sectors**
```bash
tf_env\Scripts\python.exe find_undiscovered_candidates.py
```

Produces:  
- `bls_results.csv` (validated candidates)  
- `folded_plots/` (phase‑folded BLS diagnostics)

---

## **4. Outputs**

| Component | Description |
|----------|-------------|
| `exoplanet_cnn_real.h5` | Final trained CNN model |
| `bls_results.csv` | BLS‑validated candidate periods, SNR, transit depth |
| `folded_plots/` | Diagnostic plots for each candidate |
| `confirmed_exoplanets.csv` | TAP‑retrieved confirmed host metadata |
| `*_vectors.npy` | Preprocessed training vectors |

---


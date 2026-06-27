
---

# **TESS Exoplanet CNN Pipeline**

A fully reproducible machine‑learning pipeline for detecting exoplanet transit signals in **real TESS light curves**.  
The system trains a **1D Convolutional Neural Network** on confirmed exoplanet hosts and non‑planet stars, then applies the model to unvetted TESS targets. High‑confidence predictions are validated using **Box Least Squares (BLS)** period searches.

This repository contains all scripts required to:

- build training datasets from NASA archives  
- train a real‑data CNN  
- scan new TESS sectors  
- run BLS validation  
- generate candidate plots and outputs  

---

## **Pipeline Overview**

1. **Confirmed Exoplanet Retrieval**  
   Downloads all TESS‑observed confirmed exoplanet hosts from the NASA Exoplanet Archive (TAP).

2. **Negative Sample Construction**  
   Builds a dataset of stars with no known planets or TOIs to teach the model non‑transit behavior.

3. **1D CNN Training**  
   Trains on real PDCSAP light curves to learn authentic transit morphology.

4. **Sector‑Wide Candidate Search**  
   Queries MAST for all stars in a chosen sector, removes known hosts/TOIs, and evaluates remaining targets.

5. **BLS Validation**  
   Runs BLS on any star with CNN confidence ≥0.70 to check for periodic transit‑like dips.

---

## **Architecture Diagram**

```
NASA Exoplanet Archive (TAP)
      ↓
download_confirmed_exoplanets.py → confirmed_exoplanets.csv
      ↓
generate_confirmed_dataset.py ───→ confirmed_positive_vectors.npy
generate_negative_dataset.py ────→ negative_vectors.npy
      ↓
cnn_model.py (train CNN) ───────→ exoplanet_cnn_real.h5
      ↓
find_undiscovered_candidates.py  ← MAST sector query
      ↓
(Filter known hosts & TOIs)
      ↓
(CNN prediction)
      ↓
(BLS validation) ─────────────→ bls_results.csv + folded_plots/
```

---

## **Environment Setup**

Activate the included virtual environment:

```bash
tf_env\Scripts\activate
```

### **Dependencies**
- `lightkurve` — TESS light curve access  
- `tensorflow` — CNN model  
- `astropy` — BLS period search  
- `pyvo` — TAP queries  
- `numpy`, `scipy`, `pandas`, `sklearn`, `matplotlib`

---

## **Noise Handling & Preprocessing**

The pipeline applies several preprocessing steps to ensure clean, usable light curves:

- **PDCSAP flux** for systematics‑corrected data  
- **Asymmetric outlier removal** (preserves dips, removes flares)  
- **Savitzky–Golay detrending** for long‑term variability  
- **Scatter thresholding** to exclude high‑noise targets  

---

## **Usage**

### **1. Download Confirmed Exoplanets**
```bash
tf_env\Scripts\python.exe download_confirmed_exoplanets.py
```

### **2. Generate Training Datasets**
**Positive samples:**
```bash
tf_env\Scripts\python.exe generate_confirmed_dataset.py
```

**Negative samples:**
```bash
tf_env\Scripts\python.exe generate_negative_dataset.py
```

### **3. Train the CNN**
```bash
tf_env\Scripts\python.exe cnn_model.py
```

Outputs:
- `exoplanet_cnn_real.h5`  
- `training_history_real.png`

### **4. Search for New Candidates**
```bash
tf_env\Scripts\python.exe find_undiscovered_candidates.py
```

This script:
- retrieves known hosts/TOIs  
- queries all stars in the selected sector  
- filters known objects  
- downloads + preprocesses light curves  
- runs CNN predictions  
- performs BLS validation  
- saves results + folded transit plots  

---


"""Generate a dataset of confirmed exoplanet host light curves.

This script reads 'confirmed_exoplanets.csv', extracts unique TIC IDs, and uses the
pipeline to download and standardize the TESS light curves. The resulting NumPy
array is saved as 'confirmed_positive_vectors.npy'.
"""

import csv
import numpy as np
from pipeline import process_single_target

def load_unique_tics(csv_path):
    """Return a list of unique TIC IDs from the CSV.
    The CSV is expected to have a column named 'TIC'.
    """
    tics = set()
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Skip comment lines and find the header row
        header = None
        for row in reader:
            if row and not row[0].startswith('#'):
                header = row
                break
        if header is None:
            raise ValueError("No valid header found in CSV.")
        
        # Find TIC column by exact match
        tic_col_index = None
        for i, col in enumerate(header):
            if col.strip() == 'TIC':
                tic_col_index = i
                break
        if tic_col_index is None:
            raise KeyError("Column 'TIC' not found in CSV header.")
        
        # Collect TIC values from all data rows
        for row in reader:
            if len(row) > tic_col_index and row[tic_col_index].strip():
                tics.add(row[tic_col_index].strip())
    return list(tics)

def compile_confirmed_dataset(csv_path='toi-catalog_2026-06-23.csv', output_path='confirmed_positive_vectors.npy', max_samples=None):
    tics = load_unique_tics(csv_path)
    print(f"Loaded {len(tics)} unique TOI candidates from {csv_path}")

    # Use all available candidates - no sampling limit
    max_samples = len(tics)
    vectors = []
    downloaded = 0
    print(f"Processing all {max_samples} TOI candidates for training dataset...")

    for tic in tics:
        print(f"Fetching light curve for TOI candidate TIC {tic} ({downloaded+1}/{max_samples})...")
        vec = process_single_target(tic_id=tic, sector_num=None, max_noise=0.02)
        if vec is not None:
            vectors.append(vec)
            downloaded += 1
        else:
            print(f"Warning: Could not retrieve light curve for TIC {tic}")

    if vectors:
        dataset = np.array(vectors)
        np.save(output_path, dataset)
        print(f"Saved confirmed dataset to {output_path} with shape {dataset.shape}")
    else:
        print("No light curves were successfully retrieved.")

if __name__ == "__main__":
    compile_confirmed_dataset()

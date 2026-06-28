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
        reader = csv.DictReader(f)
        for row in reader:
            tics.add(row['TIC'].strip())
    return list(tics)

def compile_confirmed_dataset(csv_path='toi-catalog_2026-06-23.csv', output_path='confirmed_positive_vectors.npy', max_samples=15):
    tics = load_unique_tics(csv_path)
    vectors = []
    downloaded = 0
    print(f"Loaded {len(tics)} unique confirmed hosts. Attempting to download up to {max_samples}...")
    
    for tic in tics:
        if downloaded >= max_samples:
            break
            
        print(f"Fetching light curve for confirmed host TIC {tic} ({downloaded+1}/{max_samples})...")
        vec = process_single_target(tic_id=tic, sector_num=None, max_noise=0.02)
        if vec is not None:
            vectors.append(vec)
            downloaded += 1
            
    if vectors:
        dataset = np.array(vectors)
        np.save(output_path, dataset)
        print(f"Saved confirmed dataset to {output_path} with shape {dataset.shape}")
    else:
        print("No light curves were successfully retrieved.")

if __name__ == "__main__":
    compile_confirmed_dataset(max_samples=1000)

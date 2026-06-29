"""Generate a dataset of negative (no known exoplanet) light curves.

Queries NASA Exoplanet Archive to identify known hosts and candidates to exclude,
queries MAST (with local CSV fallback) for observed TICs, filters out known sources,
and processes clean light curves using the pipeline.
"""

import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
import io
import sys
import os
import json
from pipeline import process_single_target

def fetch_excluded_tics():
    """Fetch all confirmed host TIC IDs and TOI candidate TIC IDs to exclude."""
    print("Fetching known planet hosts and TOI candidates from NASA Exoplanet Archive...")
    excluded = set()

    try:
        query_ps = "select distinct tic_id from ps where tic_id is not null"
        url_ps = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={urllib.parse.quote(query_ps)}&format=csv"
        req = urllib.request.Request(url_ps, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as r:
            df_ps = pd.read_csv(io.StringIO(r.read().decode('utf-8')))
            for val in df_ps['tic_id'].dropna().unique():
                val_str = str(val).split('.')[0].strip()
                if val_str.isdigit():
                    excluded.add(val_str)
            print(f"Loaded {len(df_ps)} confirmed hosts to exclude.")
    except Exception as e:
        print(f"Warning: Could not fetch confirmed hosts: {e}. Proceeding with candidate fetch...")

    try:
        # Use the local TOI catalog as the source of TOI candidates to exclude
        toi_path = 'toi-catalog_2026-06-23.csv'
        if os.path.exists(toi_path):
            with open(toi_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                # Skip header if it contains metadata
                header = next(reader)
                if 'TIC' not in header:
                    # Look for TIC column in header
                    tic_col = None
                    for col in header:
                        if 'TIC' in col or 'tic' in col:
                            tic_col = col
                            break
                    if tic_col is None:
                        print("Warning: Could not find TIC column in TOI catalog")
                        return excluded
                else:
                    tic_col = 'TIC'
                
                for row in reader:
                    val_str = str(row[tic_col]).strip()
                    if val_str.isdigit():
                        excluded.add(val_str)
            print(f"Loaded {len(excluded)} TOI candidates from local catalog to exclude.")
        else:
            # Fallback to remote TOI query if local file not available
            query_toi = "select distinct tid from toi where tid is not null"
            url_toi = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={urllib.parse.quote(query_toi)}&format=csv"
            req = urllib.request.Request(url_toi, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as r:
                df_toi = pd.read_csv(io.StringIO(r.read().decode('utf-8')))
                for val in df_toi['tid'].dropna().unique():
                    val_str = str(val).split('.')[0].strip()
                    if val_str.isdigit():
                        excluded.add(val_str)
                print(f"Loaded {len(df_toi)} TOI candidates from NASA archive to exclude.")
    except Exception as e:
        print(f"Warning: Could not fetch TOI candidates from local catalog or remote: {e}.")
        
    print(f"Total unique TIC IDs to exclude: {len(excluded)}")
    return excluded

def fetch_mast_targets(sector=21):
    """Fetch TESS targets observed in a sector using astroquery.mast."""
    from astroquery.mast import Observations
    tics = []

    try:
        print(f"[1/3] Querying MAST for TESS Sector {sector} targets using astroquery...")
        observations = Observations.query_criteria(
            obs_collection="TESS",
            dataproduct_type="timeseries",
            sequence_number=sector
        )

        print(f"   astroquery returned {len(observations)} rows")
        if len(observations) > 0:
            print(f"   Available columns: {list(observations.columns)}")
            for i, row in enumerate(observations[:5]):
                target_name = row.get('target_name', '')
                print(f"   Row {i}: target_name='{target_name}' (type: {type(target_name)})")

        if len(observations) > 0:
            for row in observations:
                target_name = row.get('target_name', '')
                if isinstance(target_name, str):
                    if target_name.startswith('TIC '):
                        tic_id = target_name[4:].strip()
                        if tic_id.isdigit():
                            tics.append(tic_id)
                    elif target_name.isdigit():
                        tics.append(target_name)

        print(f"[3/3] Done. Retrieved {len(tics)} numeric TIC targets from MAST for Sector {sector}.")
    except Exception as e:
        print(f"Warning: astroquery failed: {e}")

    if not tics:
        print(f"[1/3] Building MAST API request for Sector {sector}...")
        request = {
            "service": "Mast.Caom.Cone",
            "params": {
                "ra": 0.0,
                "dec": 0.0,
                "radius": 180.0,
                "obs_collection": "TESS",
                "dataproduct_type": "timeseries",
                "sequence_number": sector
            },
            "format": "json",
            "pagesize": 10000,
            "page": 1
        }

        request_json = json.dumps(request)
        request_encoded = urllib.parse.quote(request_json)
        url = "https://mast.stsci.edu/api/v0/invoke"

        print(f"[2/3] Sending HTTP request to MAST API...")
        try:
            headers = {
                "Content-type": "application/x-www-form-urlencoded",
                "Accept": "text/plain",
                "User-Agent": "Mozilla/5.0"
            }
            data = "request=" + request_encoded
            req = urllib.request.Request(url, data=data.encode('utf-8'), headers=headers)
            with urllib.request.urlopen(req, timeout=300) as r:
                response = r.read().decode('utf-8')
                data_json = json.loads(response)

            if data_json['status'] != 'COMPLETE':
                raise Exception(f"MAST query status: {data_json['status']}")

            print(f"[3/3] Parsed {len(data_json['data'])} rows. Extracting TIC IDs...")
            if len(data_json['data']) > 0:
                print(f"   Available columns: {list(data_json['data'][0].keys())}")
                for i, row in enumerate(data_json['data'][:3]):
                    print(f"   Row {i}: target_name='{row.get('target_name', '')}'")
            for row in data_json['data']:
                target_name = row.get('target_name', '')
                if isinstance(target_name, str):
                    if target_name.startswith('TIC '):
                        tic_id = target_name[4:].strip()
                        if tic_id.isdigit():
                            tics.append(tic_id)
                    elif target_name.isdigit():
                        tics.append(target_name)
            print(f"Done. Retrieved {len(tics)} numeric TIC targets from MAST for Sector {sector}.")
        except Exception as e:
            print(f"ERROR: MAST API query failed: {e}")

    return tics

def compile_negative_dataset(output_path='negative_vectors.npy', max_samples=None, sector=21):
    excluded = fetch_excluded_tics()
    all_targets = fetch_mast_targets(sector=sector)

    if not all_targets:
        print("Error: No targets retrieved from MAST or local backup. Cannot compile negatives.", file=sys.stderr)
        sys.exit(1)

    clean_targets = [t for t in all_targets if t not in excluded]
    print(f"Filtered out {len(all_targets) - len(clean_targets)} targets as confirmed/TOI. {len(clean_targets)} remaining.")

    # Use all available clean targets as negative samples
    max_samples = len(clean_targets)
    vectors = []
    downloaded_count = 0

    print(f"Processing all {max_samples} clean control star candidates for negative dataset...")
    for tic in clean_targets:
        print(f"\nProcessing negative sample {downloaded_count+1}/{max_samples} (TIC {tic})...")
        vec = process_single_target(tic_id=tic, sector_num=sector, max_noise=0.02)
        if vec is not None:
            vectors.append(vec)
            downloaded_count += 1
        else:
            print(f"Warning: Could not retrieve light curve for TIC {tic}")

    if vectors:
        dataset = np.array(vectors)
        np.save(output_path, dataset)
        print(f"Saved negative dataset to {output_path} with shape {dataset.shape}")
    else:
        print("No negative light curves were successfully retrieved.")
            
    if vectors:
        dataset = np.array(vectors)
        np.save(output_path, dataset)
        print(f"\nSaved negative dataset to '{output_path}' with shape {dataset.shape}")
    else:
        print("No light curves were successfully retrieved for negative controls.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    compile_negative_dataset()

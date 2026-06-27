

import os
import csv
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
import tensorflow as tf
from astropy.timeseries import BoxLeastSquares
import matplotlib.pyplot as plt
import io
import sys

from pipeline import process_single_target

def fetch_known_systems():
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
        print(f"Warning: Could not fetch confirmed hosts: {e}")
        
    try:
        query_toi = "select distinct tid from toi where tid is not null"
        url_toi = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={urllib.parse.quote(query_toi)}&format=csv"
        req = urllib.request.Request(url_toi, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as r:
            df_toi = pd.read_csv(io.StringIO(r.read().decode('utf-8')))
            for val in df_toi['tid'].dropna().unique():
                val_str = str(val).split('.')[0].strip()
                if val_str.isdigit():
                    excluded.add(val_str)
        print(f"Loaded {len(df_toi)} TOI candidates to exclude.")
    except Exception as e:
        print(f"Warning: Could not fetch TOI candidates: {e}")
        
    print(f"Total unique known TIC IDs to exclude: {len(excluded)}")
    return excluded

def fetch_mast_sector_targets(sector=21):
    tics = []
    
    print(f"[1/4] Building MAST sync TAP query for Sector {sector}...")
    query = (
        f"SELECT DISTINCT target_name "
        f"FROM dbo.caomobservation "
        f"JOIN ivoa.obscore ON dbo.caomobservation.observationID = ivoa.obscore.obs_id "
        f"WHERE obs_collection = 'TESS' "
        f"AND dataproduct_type = 'timeseries' "
        f"AND sequenceNumber = {sector}"
    )
    params = urllib.parse.urlencode({
        'REQUEST': 'doQuery',
        'LANG': 'ADQL',
        'QUERY': query,
        'FORMAT': 'csv'
    })
    url = f"https://mast.stsci.edu/vo-tap/api/v0.1/caom/sync?{params}"
    
    print(f"[2/4] Sending HTTP request to MAST (no timeout — may take a while)...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as r:
            print(f"[3/4] Response received! Reading and parsing CSV data...")
            raw = r.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(raw))
            
        print(f"[4/4] Parsed {len(df)} rows. Extracting TIC IDs...")
        for val in df['target_name'].dropna().unique():
            val_str = str(val).strip()
            if val_str.startswith('TIC '):
                val_str = val_str[4:]
            if val_str.isdigit():
                tics.append(val_str)
        print(f"Done. Retrieved {len(tics)} numeric TIC targets from MAST for Sector {sector}.")
    except Exception as e:
        print(f"ERROR: MAST sync TAP query failed: {e}")
        
    return tics


def run_bls_validation(time, flux, period_range=(0.5, 20.0)):
    bls = BoxLeastSquares(time, flux)
    periods = np.linspace(period_range[0], period_range[1], 1000)
    result = bls.power(periods, 0.05, 0.3)
    
    max_idx = np.argmax(result.power)
    best_period = float(result.period[max_idx])
    best_depth = float(result.depth[max_idx])
    best_duration = float(result.duration[max_idx])
    best_power = float(result.power[max_idx])
    best_transit_time = float(result.transit_time[max_idx])
    
    return {
        'period': best_period,
        'depth': best_depth,
        'duration': best_duration,
        'power': best_power,
        'transit_time': best_transit_time,
        'bls_model': bls,
        'bls_result': result
    }

def screen_candidates(sector=21, max_candidates=50, prediction_threshold=0.7):
    model_path = 'exoplanet_cnn_real.h5'
    if not os.path.exists(model_path):
        print(f"Warning: Real model '{model_path}' not found. Falling back to synthetic model 'exoplanet_cnn_model.h5'.")
        model_path = 'exoplanet_cnn_model.h5'
        if not os.path.exists(model_path):
            print("Error: No trained CNN model found. Please run cnn_model.py first.", file=sys.stderr)
            sys.exit(1)
            
    print(f"Loading trained CNN model from '{model_path}'...")
    model = tf.keras.models.load_model(model_path)
    
    excluded = fetch_known_systems()
    
    all_targets = fetch_mast_sector_targets(sector=sector)
    if not all_targets:
        print("Error: No target stars retrieved from MAST or local backup.", file=sys.stderr)
        sys.exit(1)
        
    uncovered_targets = [t for t in all_targets if t not in excluded]
    print(f"Filtered out {len(all_targets) - len(uncovered_targets)} targets already covered by NASA.")
    print(f"{len(uncovered_targets)} unanalyzed targets remaining.")
    
    with open('candidate_stars.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tic', 'sector'])
        for tic in uncovered_targets:
            writer.writerow([tic, sector])
    print(f"Saved {len(uncovered_targets)} unanalyzed targets to 'candidate_stars.csv'.")
    
    results = []
    screened_count = 0
    candidate_count = 0
    
    print(f"\nScreening up to {max_candidates} targets with CNN...")
    for tic in uncovered_targets:
        if screened_count >= max_candidates:
            break
            
        print(f"\n--- Screening target {screened_count+1}/{max_candidates} (TIC {tic}) ---")
        flux = process_single_target(tic_id=tic, sector_num=sector, max_noise=0.02)
        screened_count += 1
        
        if flux is None:
            print("Skipping target (failed download or too noisy).")
            continue
            
        X_input = flux.reshape(1, -1, 1)
        prediction = float(model.predict(X_input, verbose=0)[0][0])
        print(f"CNN Exoplanet Confidence Score: {prediction:.4f}")
        
        if prediction >= prediction_threshold:
            print(f"🔥 CNN Candidate Detected (Confidence {prediction:.4f} >= {prediction_threshold})!")
            candidate_count += 1
            
            time = np.linspace(0, len(flux) * 0.020833, len(flux))
            
            print("Running Box Least Squares (BLS) period analysis...")
            bls_res = run_bls_validation(time, flux)
            print(f"BLS Best Period: {bls_res['period']:.4f} days, Depth: {bls_res['depth']:.5f}, Power: {bls_res['power']:.2f}")
            
            results.append({
                'tic': tic,
                'sector': sector,
                'cnn_confidence': prediction,
                'bls_period': bls_res['period'],
                'bls_depth': bls_res['depth'],
                'bls_duration': bls_res['duration'],
                'bls_power': bls_res['power'],
                'bls_transit_time': bls_res['transit_time'],
                'flux': flux,
                'time': time,
                'bls_info': bls_res
            })
            
    if results:
        with open('bls_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'tic', 'sector', 'cnn_confidence', 'bls_period', 'bls_depth', 
                'bls_duration', 'bls_power', 'bls_transit_time'
            ])
            for r in results:
                writer.writerow([
                    r['tic'], r['sector'], r['cnn_confidence'], r['bls_period'], 
                    r['bls_depth'], r['bls_duration'], r['bls_power'], r['bls_transit_time']
                ])
        print(f"\nScreening completed! Found {len(results)} new candidates.")
        print("Results saved to 'bls_results.csv'.")
        
        top_candidates = sorted(results, key=lambda x: x['bls_power'], reverse=True)[:3]
        for i, cand in enumerate(top_candidates):
            plt.figure(figsize=(10, 4))
            
            t_fold = (cand['time'] - cand['bls_transit_time'] + 0.5 * cand['bls_period']) % cand['bls_period'] - 0.5 * cand['bls_period']
            
            plt.scatter(t_fold, cand['flux'], color='royalblue', s=3, alpha=0.5, label='Data')
            
            bin_size = 0.05
            bins = np.arange(-0.5 * cand['bls_period'], 0.5 * cand['bls_period'], bin_size)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            binned_flux = []
            for b_start, b_end in zip(bins[:-1], bins[1:]):
                mask = (t_fold >= b_start) & (t_fold < b_end)
                if np.any(mask):
                    binned_flux.append(np.mean(cand['flux'][mask]))
                else:
                    binned_flux.append(np.nan)
                    
            plt.plot(bin_centers, binned_flux, color='red', linewidth=2, label='Binned Average')
            
            plt.title(f"Folded Light Curve for Undiscovered Candidate TIC {cand['tic']} (P={cand['bls_period']:.4f}d)")
            plt.xlabel("Phase [days]")
            plt.ylabel("Normalized Flux")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            plot_path = f"undiscovered_candidate_TIC_{cand['tic']}.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved diagnostic folded plot as '{plot_path}'")
    else:
        print("\nScreening completed. No candidates met the prediction threshold.")
        
if __name__ == "__main__":
    screen_candidates(sector=21, max_candidates=15)

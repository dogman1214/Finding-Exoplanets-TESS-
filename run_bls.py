import numpy as np
from scipy.signal import find_peaks
from astropy.timeseries import BoxLeastSquares
import matplotlib.pyplot as plt
import csv

def run_box_least_squares(time, flux, period_range=(0.5, 20.0), min_duration=0.1, max_duration=0.5):
    
    bls = BoxLeastSquares(time, flux)
    
    periods = np.linspace(period_range[0], period_range[1], 1000)
    
    result = bls.power(periods, min_duration, max_duration)
    
    peaks, properties = find_peaks(result.power, height=np.percentile(result.power, 95))
    
    top_peaks = np.argsort(result.power[peaks])[::-1][:5]
    
    candidates = []
    for i in top_peaks:
        peak_idx = peaks[i]
        candidates.append({
            'period': float(result.period[peak_idx]),
            'depth': float(result.depth[peak_idx]),
            'duration': float(result.duration[peak_idx]),
            'power': float(result.power[peak_idx]),
            'transit_time': float(result.transit_time[peak_idx])
        })
    
    return candidates, result

def analyze_cnn_candidates(cnn_results_csv='cnn_results.csv', targets_csv='targets.csv', output_csv='bls_results.csv'):
    
    import csv
    cnn_results = []
    
    with open(cnn_results_csv, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            cnn_results.append({
                'tic': row['tic'],
                'sector': int(row['sector']),
                'confidence': float(row['confidence'])
            })
    
    high_confidence = [r for r in cnn_results if r['confidence'] > 0.7]
    
    print(f"Running BLS on {len(high_confidence)} high-confidence CNN candidates...")
    
    bls_results = []
    
    for candidate in high_confidence:
        tic = candidate['tic']
        sector = candidate['sector']
        confidence = candidate['confidence']
        
        light_curve = process_single_target(tic, sector)
        
        if light_curve is None:
            continue
        
        time = np.linspace(0, len(light_curve)*0.020833, len(light_curve))
        
        candidates, result = run_box_least_squares(time, light_curve)
        
        for bls_candidate in candidates:
            bls_results.append({
                'tic': tic,
                'sector': sector,
                'cnn_confidence': confidence,
                'bls_period': bls_candidate['period'],
                'bls_depth': bls_candidate['depth'],
                'bls_duration': bls_candidate['duration'],
                'bls_power': bls_candidate['power'],
                'bls_transit_time': bls_candidate['transit_time']
            })
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'tic', 'sector', 'cnn_confidence', 'bls_period', 'bls_depth', 
            'bls_duration', 'bls_power', 'bls_transit_time'
        ])
        for result in bls_results:
            writer.writerow([
                result['tic'], 
                result['sector'], 
                result['cnn_confidence'], 
                result['bls_period'], 
                result['bls_depth'], 
                result['bls_duration'], 
                result['bls_power'], 
                result['bls_transit_time']
            ])
    
    print(f"\nBLS analysis complete!")
    print(f"Total candidates analyzed: {len(high_confidence)}")
    print(f"Total BLS transit signals detected: {len(bls_results)}")
    print(f"Results saved to {output_csv}")
    
    return bls_results

def process_single_target(tic_id, sector_num):
    import lightkurve as lk
    import numpy as np
    from scipy.interpolate import interp1d
    
    try:
        print(f"Connecting to NASA servers to fetch TIC {tic_id} (Sector {sector_num})...")
        
        search_query = lk.search_lightcurve(f"TIC {tic_id}", sector=sector_num, author="SPOC")
        if len(search_query) == 0:
            print(f"Warning: Target TIC {tic_id} not available in Sector {sector_num}.")
            return None
        
        light_curve = search_query[0].download()
        
        cleaned_lc = light_curve.remove_nans().remove_outliers(sigma=5)
        
        flattened_lc = cleaned_lc.flatten(window_length=101)
        
        time_points = flattened_lc.time.value
        flux_values = flattened_lc.flux.value
        
        uniform_time_grid = np.linspace(time_points.min(), time_points.max(), 2000)
        interpolation_engine = interp1d(time_points, flux_values, kind="linear", fill_value="extrapolate")
        standardized_flux = interpolation_engine(uniform_time_grid)
        
        min_val = np.min(standardized_flux)
        max_val = np.max(standardized_flux)
        normalized_flux = (standardized_flux - min_val) / (max_val - min_val)
        
        return normalized_flux
        
    except Exception as error_logs:
        print(f"Pipeline crashed on target TIC {tic_id}: {error_logs}")
        return None

if __name__ == "__main__":
    import os
    if not os.path.exists('cnn_results.csv'):
        print("Running CNN analysis first...")
        from run_cnn_analysis import analyze_catalog_from_csv
        analyze_catalog_from_csv('targets.csv')
    
    bls_results = analyze_cnn_candidates()
    
    if len(bls_results) > 0:
        print("\nTop BLS Candidates:")
        for i, result in enumerate(sorted(bls_results, key=lambda x: x['bls_power'], reverse=True)[:5]):
            print(f"{i+1}. TIC {result['tic']} (S{result['sector']}): P={result['bls_period']:.3f}d, "
                  f"depth={result['bls_depth']:.5f}, power={result['bls_power']:.3f}")
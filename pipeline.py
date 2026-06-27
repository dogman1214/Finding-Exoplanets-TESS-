import lightkurve as lk
import numpy as np
from scipy.interpolate import interp1d

def process_single_target(tic_id, sector_num=None, target_length=2000, max_noise=0.02, window_length=101):
    """Process a single TIC ID.
    
    Parameters
    ----------
    tic_id : str
        TIC identifier.
    sector_num : int or None, optional
        Specific TESS sector to search. If None, searches all available sectors.
    target_length : int, optional
        Length of the output standardized flux array.
    max_noise : float, optional
        Maximum allowed standard deviation of the detrended flux. Targets exceeding
        this will be rejected as too noisy. Defaults to 0.02 (2% scatter).
    window_length : int, optional
        The Savitzky-Golay filter window size for detrending/flattening.
    """
    try:
        print(f"Connecting to NASA/MAST servers to fetch TIC {tic_id} (Sector {sector_num})...")
        
        search_query = None
        for author in ["SPOC", "QLP"]:
            if sector_num is not None:
                search_query = lk.search_lightcurve(f"TIC {tic_id}", sector=sector_num, author=author)
            else:
                search_query = lk.search_lightcurve(f"TIC {tic_id}", author=author)
            if len(search_query) > 0:
                print(f"Found TESS data from author: {author}")
                break
                
        if search_query is None or len(search_query) == 0:
            print(f"Warning: Target TIC {tic_id} not available in Sector {sector_num}.")
            return None
        
        print(f"Downloading light curve for TIC {tic_id}...")
        light_curve = search_query[0].download()
        
        print(f"Cleaning light curve for TIC {tic_id}...")
        if 'pdcsap_flux' in light_curve.columns:
            light_curve.flux = light_curve.pdcsap_flux
            
        cleaned_lc = light_curve.remove_nans().remove_outliers(sigma_upper=4, sigma_lower=10)
        
        print(f"Flattening light curve for TIC {tic_id} using window_length={window_length}...")
        flattened_lc = cleaned_lc.flatten(window_length=window_length)
        
        time_points = flattened_lc.time.value
        flux_values = flattened_lc.flux.value
        
        noise_level = np.std(flux_values)
        print(f"Estimated noise level (scatter) for TIC {tic_id}: {noise_level:.5f}")
        
        if noise_level > max_noise:
            print(f"REJECTED: TIC {tic_id} noise level ({noise_level:.5f}) exceeds maximum threshold ({max_noise}).")
            return None
            
        uniform_time_grid = np.linspace(time_points.min(), time_points.max(), target_length)
        interpolation_engine = interp1d(time_points, flux_values, kind="linear", fill_value="extrapolate")
        standardized_flux = interpolation_engine(uniform_time_grid)
        
        min_val = np.min(standardized_flux)
        max_val = np.max(standardized_flux)
        normalized_flux = (standardized_flux - min_val) / (max_val - min_val)
        
        print(f"Successfully processed TIC {tic_id} (Sector {sector_num}) with {len(normalized_flux)} data points")
        return normalized_flux
        
    except Exception as error_logs:
        print(f"ERROR: Pipeline crashed on target TIC {tic_id}: {str(error_logs)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    pass
"""
Uses BATMAN to make synthetic light curves of exoplanet transits 

It creates realistic light curves with: 
- Stellar limb darkening
- Orbital ecentricity
- Gaussian noise (fancy)
-Stellar variability (basically red noise)
- Instrumental offsets

then saves as an npy file so our CNN can ingest it
"""


import numpy as np
import batman #oooh yeah I feel like a superhero
import matplotlib.pyplot as plt
from scipy import ndimage
import random #huh


def generate_synthetic_transit(params, t, limb_dark="quadratic"):
    """
    builds one measely light curve using BATMAN. This will run like tens of thousands of times in order to make a full dataset if not millions
    
    params: batman.Transitparams object with transit parameters
    t: array of time points (days)
    limb_dark: limb darkening law (uniform, inear, quadratic, nonlinear)


    returns the flux, a normalized flux array.
    """

    m = batman.TransitModel(params, t)

    flux = m.light_curve(params)

    return flux


def add_stellar_variability(flux, t, variability_amplitude = 0.001, variability_timescale = 1.0):
    """
    red noise stuff
    """

    noise_length = len(t)
    dt = np.median(np.diff(t))
    timescale_samples = int(variability_timescale / dt)


    white_noise = np.random.normal(0, variability_amplitude, noise_length)

    if timescale_samples > 1:
        kernel = np.ones(timescale_samples) / timescale_samples
        red_noise = np.convolve(white_noise, kernel, mode='same')
    else:
        red_noise = white_noise

    flux_with_variability = flux + red_noise   #flux capacitor type stuff (jk)

    flux_with_variability = np.clip(flux_with_variability, 0.9, 1.1)

    return flux_with_variability

def add_gaussian_noise(flux, noise_level = 0.0005)
    """
    does what it says
    are you serious
    """

    noise = np.random.normal(0, noise_level, len(flux))
    flux_with_noise = flux + noise
    return flux_with_noise

def add_instrumental_offset(flux, offset_mean=0, offset_std = 0.0001):

    """
    supposed to simulate different observing conditions
    """

    offset = np.random.normal(offset_mean, offset_std)
    flux_with_offset = flux + offset
    return flux_with_offset

def generate_single_light_curve(tic_id=None, period=1.0, duration=0.1, depth=0.01, eccentricity=0.0, omega = 0.0, limb_dark_coeff= [0.3, 0.3], noise_level = 0.0005, variability_amplitude = 0.001, t_length = 15, cadence = 0.02):

    # makes complete synthetic light curve

    t = np.arange(0, t_length, cadence)

    params = batman.TransitParams()

    params.t0 = t_length / 2.0 # time of inferior conjunction
    params.per = period #orbital period
    params.rp = np.sqrt(depth) #planet radius
    params.a = 13.5 #semi-amjor axis
    params.inc = 89.0 #inclination (degrees)
    params.ecc = eccentricity
    params.w = omega #lonitude of periastron
    params.limb_dark = "quadratic"
    params.u = limb_dark_coeff

    flux = generate_synthetic_transit(params, t)

    flux = add_stellar_variability(flux, t, variability_amplitude, variability_timescale = 1.0)

    flux = add_gaussian_noise(flux, noise_level)

    flux = add_instrumental_offset(flux)

    flux = flux / np.mean(flux)

    metadata = {
        'tic_id': tic_id if tic_id else f"SYN_{random.randint(10000000, 99999999)}",
        'period': period,
        'duration': duration,
        'depth': depth,
        'eccentricity': eccentricity,
        'omega': omega,
        'limb_dark_coeff': limb_dark_coeff,
        'noise_level': noise_level,
        'variability_amplitude': variability_amplitude,
        't_length': t_length,
        'cadence': cadence
    }
    
    return flux, metadata

def generate_synthetic_dataset(n_samples = 1000,
                                output_path='synthetic_positive_vectors.npy',
                                min_period=0.5, 
                                max_period=15.0,
                                min_depth=0.0001, 
                                max_depth=0.05,
                                min_duration=0.05, 
                                max_duration=0.3,
                                min_eccentricity=0.0,
                                max_eccentricity=0.3):
                    
    """
    generates the complete dataset

    Parameters:
    - n_samples: number of synthetic light curves to generate
    - output_path: where to save the .npy file
    - min/max parameters: range for random sampling of transit parameters
    """

    print(f"Generating {n_samples} synthetic light curves...")

    dataset = []
    metadata_list = []


    for i in range(n_samples):
        period = np.random.uniform(min_period, max_period)
        duration = np.random.uniform(min_duration, max_duration)
        depth = np.random.uniform(min_depth, max_depth)
        eccentricity = np.random.uniform(min_eccentricity, max_eccentricity)
        omega = np.random.uniform(0, 2*np.pi)
        
        # Generate limb darkening coefficients (quadratic) from realistic ranges
        u1 = np.random.uniform(0.1, 0.5)
        u2 = np.random.uniform(0.1, 0.5)
        limb_dark_coeff = [u1, u2]
        
        # Generate the light curve
        flux, metadata = generate_single_light_curve(
            period=period,
            duration=duration,
            depth=depth,
            eccentricity=eccentricity,
            omega=omega,
            limb_dark_coeff=limb_dark_coeff,
            noise_level=np.random.uniform(0.0002, 0.001),
            variability_amplitude=np.random.uniform(0.0002, 0.002)
        )
        
        dataset.append(flux)
        metadata_list.append(metadata)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i+1}/{n_samples} light curves...")
    
    dataset_array = np.array(dataset)
    
    # Save dataset
    np.save(output_path, dataset_array)
    
    print(f"\nDataset generation complete!")
    print(f"Saved synthetic dataset to {output_path} with shape {dataset_array.shape}")
    print(f"Dataset contains {len(dataset_array)} synthetic light curves")
    
    return dataset_array, metadata_list

if __name__ == "__main__":
    generate_synthetic_dataset(
        n_samples=5000,  
        output_path='synthetic_positive_vectors.npy',
        min_period=0.5,
        max_period=15.0,
        min_depth=0.0001,
        max_depth=0.05,
        min_duration=0.05,
        max_duration=0.3
    )   

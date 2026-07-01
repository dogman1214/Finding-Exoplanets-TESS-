

import numpy as np
import random
import sys

def generate_synthetic_negative_light_curve(length=2000):
    
    
    noise_level = np.random.uniform(0.005, 0.015)
    
    noise = np.random.normal(0, noise_level, length)
    
   
    dt = 0.02  
    timescale = np.random.uniform(0.5, 2.0)  
    timescale_samples = int(timescale / dt)
    
    if timescale_samples > 1:
        kernel = np.ones(timescale_samples) / timescale_samples
        red_noise = np.convolve(noise, kernel, mode='same')
        noise = red_noise
    
    trend_strength = np.random.uniform(0.0001, 0.0005)
    trend = np.linspace(0, trend_strength, length)
    
    flux = 1.0 + noise + trend
    
    flux = np.clip(flux, 0.9, 1.1)
    
    flux = flux / np.mean(flux)
    
    flux = flux.reshape(-1, 1)
    
    return flux

def generate_synthetic_negative_dataset(n_samples=20000, output_path='synthetic_negative_vectors.npy'):
   
    
    print(f"Generating {n_samples} synthetic negative (noise-only) light curves...")
    
    dataset = []
    
    for i in range(n_samples):
        try:
            flux = generate_synthetic_negative_light_curve()
            dataset.append(flux)
            
            if (i + 1) % 1000 == 0:
                print(f"Generated {i+1}/{n_samples} synthetic negative samples...")
                
        except Exception as e:
            print(f"Error generating synthetic negative sample {i}: {e}")
            continue
    
    if dataset:
        dataset_array = np.array(dataset)
        np.save(output_path, dataset_array)
        print(f"\nSynthetic negative dataset generation complete!")
        print(f"Saved to {output_path} with shape {dataset_array.shape}")
        return dataset_array
    else:
        print("No synthetic negative samples were generated.")
        return None

if __name__ == "__main__":
    # Generate 20,000 synthetic negative samples
    generate_synthetic_negative_dataset(
        n_samples=20000,
        output_path='synthetic_negative_vectors.npy'
    )
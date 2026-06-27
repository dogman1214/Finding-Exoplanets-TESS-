import numpy as np
import tensorflow as tf
from pipeline import process_single_target
import matplotlib.pyplot as plt

def load_model_and_predict(model_path='exoplanet_cnn_model.h5'):
    '''Load trained model and make predictions on TESS data'''
    
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")
    
    return model

def analyze_target_with_cnn(tic_id, sector_num, model):
    '''Analyze a single TESS target using the trained CNN model'''
    
    print(f"Processing TIC {tic_id} (Sector {sector_num})...")
    light_curve = process_single_target(tic_id, sector_num)
    
    if light_curve is None:
        print(f"WARNING: Could not retrieve data for TIC {tic_id} (Sector {sector_num})")
        return None
    
    X_input = light_curve.reshape(1, -1, 1)
    
    prediction = model.predict(X_input)[0][0]
    
    return {
        'tic_id': tic_id,
        'sector': sector_num,
        'prediction': float(prediction),
        'confidence': float(prediction),
        'light_curve': light_curve
    }

def analyze_catalog_from_csv(csv_path, model_path='exoplanet_cnn_model.h5', output_csv='cnn_results.csv'):
    '''Analyze all targets in a CSV file using the CNN model'''
    
    model = load_model_and_predict(model_path)
    
    import csv
    targets = []
    
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            targets.append({
                'tic': row['tic'].strip(),
                'sector': int(row['sector'].strip())
            })
    
    results = []
    high_confidence_candidates = []
    
    print(f"Analyzing {len(targets)} targets with CNN model...")
    
    for target in targets:
        tic = target['tic']
        sector = target['sector']
        
        result = analyze_target_with_cnn(tic, sector, model)
        if result is not None:
            results.append(result)
            
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['tic', 'sector', 'prediction', 'confidence'])
        for result in results:
            writer.writerow([
                result['tic_id'], 
                result['sector'], 
                result['prediction'], 
                result['confidence']
            ])
    
    print(f"\nAnalysis complete!")
    print(f"Total targets analyzed: {len(results)}")
    print(f"Results saved to {output_csv}")
    
    return results, high_confidence_candidates

if __name__ == "__main__":
    results, high_confidence = analyze_catalog_from_csv('targets.csv')
    
    if len(high_confidence) > 0:
        fig, axes = plt.subplots(min(3, len(high_confidence)), 1, figsize=(12, 4*min(3, len(high_confidence))))
        if len(high_confidence) == 1:
            axes = [axes]
        
        for i, candidate in enumerate(high_confidence[:3]):
            axes[i].plot(candidate['light_curve'])
            axes[i].set_title(f'TIC {candidate["tic_id"]}, Sector {candidate["sector"]}, Confidence: {candidate["confidence"]:0.3f}')
            axes[i].set_xlabel('Time Bins')
            axes[i].set_ylabel('Normalized Flux')
            
        plt.tight_layout()
        plt.savefig('high_confidence_candidates.png')
        print("High-confidence candidate plots saved as 'high_confidence_candidates.png'")
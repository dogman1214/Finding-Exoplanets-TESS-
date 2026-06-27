import csv
import numpy as np
from pipeline import process_single_target

def compile_dataset_from_csv(input_csv_path, output_matrix_path, tic_column='tic'):
    """Compile dataset using specified TIC column name.

    Parameters
    ----------
    input_csv_path : str
        Path to the CSV containing TIC IDs and sector numbers.
    output_matrix_path : str
        Path where the resulting NumPy array will be saved.
    tc_column : str, optional
        Name of the column in the CSV that contains the TIC ID. Defaults to 'tic'.
    """
    matrix_rows = []

    print(f"Opening data catalog source: {input_csv_path}")

    with open(input_csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            tic = row[tic_column].strip()
            sector = int(row['sector'].strip()) if 'sector' in row else None

            if sector is None:
                vector = process_single_target(tic_id=tic, sector_num=None)
            else:
                vector = process_single_target(tic_id=tic, sector_num=sector)

            if vector is not None:
                matrix_rows.append(vector)
                print(f"Logged dynamic target vector for TIC {tic} inside memory buffer.")

    if len(matrix_rows) > 0:
        dataset_matrix = np.array(matrix_rows)
        np.save(output_matrix_path, dataset_matrix)
        print(f"\nMatrix generation complete!")
        print(f"File Saved: {output_matrix_path} | Total Data Matrix Shape: {dataset_matrix.shape}")
    else:
        print("\nWarning: No valid arrays were collected from the input file source.")

if __name__ == "__main__":
    compile_dataset_from_csv(
        input_csv_path="targets.csv",
        output_matrix_path="positive_vectors.npy"
    )

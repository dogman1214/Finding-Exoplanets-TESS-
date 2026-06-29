import csv
import numpy as np
from pipeline import process_single_target

def compile_dataset_from_csv(input_csv_path, output_matrix_path, tic_column='tic'):
    matrix_rows = []

    print(f"Opening data catalog source: {input_csv_path}")

    with open(input_csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            tic = row[tic_column].strip()
            # Handle both 'sector' and 'Sectors' column names
            sector_str = row.get('sector', row.get('Sectors', ''))
            sector = None
            if sector_str and sector_str.strip():
                try:
                    # Handle both single sector and list of sectors
                    if sector_str.startswith('['):
                        # Parse list format like "[28, 68]"
                        sector_list = sector_str.strip('[]').split(',')
                        # Use first sector for consistency
                        sector = int(sector_list[0].strip())
                    else:
                        sector = int(sector_str.strip())
                except ValueError:
                    pass  # Keep sector as None if parsing fails

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
    # Use TOI catalog as source for positive examples
    compile_dataset_from_csv(
        input_csv_path="toi-catalog_2026-06-23.csv",
        output_matrix_path="positive_vectors.npy",
        tic_column='TIC'
    )

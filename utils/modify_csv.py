import os
import csv
from pathlib import Path

def modify_csv_column(file_path, target_column, old_root, new_root):
    with open(file_path, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    if rows:
        for row in rows[1:]:  # Skip the header
            if len(row) > column_index and row[column_index].startswith(old_root):
                row[column_index] = row[column_index].replace(old_root, new_root, 1)

    # Write back to the SAME file
    with open(file_path, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)

def process_csv_files_in_folder(parent_folder, target_column, old_root, new_root):
    """
    Iterate all csv files in parent_folder
    """
    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                modify_csv_column(file_path, target_column, old_root, new_root)
                print(f"Processing file: {file_path}")

if __name__ == '__main__':
    parent_folder_path = 'perturbationdrive/logs/RoadGenerator'
    column_index = 5
    old_root_path = '/home/jiaqq/Project-1120/PerturbationDrive/udacity/perturb_logs/roadGen'
    new_root_path = '/home/jiaqq/Documents/ThirdEye-II/perturbationdrive/logs/RoadGenerator'

    process_csv_files_in_folder(parent_folder_path, column_index, old_root_path, new_root_path)
import os
import pandas as pd

"""

    Author: Teng Ouyang
    Updated date: 2024-11
    Description: Load data
    GitHub: https://github.com/TyreseOu/AADS

"""


def load_data(target_folder, source_folder, ground_truth_file, target_project, source_project):
    ground_truth_df = pd.read_excel(ground_truth_file, sheet_name=None)

    all_path_target, all_labels_target = [], []

    target_ground_truth_sheet = ground_truth_df.get(target_project)
    if target_ground_truth_sheet is not None:
        for index, row in target_ground_truth_sheet.iterrows():
            file_contract = row['contract']
            file = row['file']
            target_file = os.path.join(target_folder, f"{file}_{file_contract}.sol")

            if os.path.exists(target_file):
                all_path_target.append(target_file)
                all_labels_target.append(row['ground truth'])
            else:
                continue

        if source_project != target_project:
            all_path_source, all_labels_source = [], []
            source_ground_truth_sheet = ground_truth_df.get(source_project)
            for index, row in source_ground_truth_sheet.iterrows():
                file_contract = row['contract']
                file = row['file']
                source_file = os.path.join(source_folder, f"{file}_{file_contract}.sol")

                if os.path.exists(source_file):
                    all_path_source.append(source_file)
                    all_labels_source.append(row['ground truth'])
                else:
                    continue

            return (
                all_path_target,
                all_labels_target,
                all_path_source,
                all_labels_source,
            )

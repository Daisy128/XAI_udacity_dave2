import os

import pandas as pd

heatmap_type_list = ["smooth_grad", "raw_smooth_grad",
                     "grad_cam_pp", "raw_grad_cam_pp",
                     "faster_score_cam", "raw_faster_score_cam",
                     "integrated_gradients", "raw_integrated_gradients", ]

focus = "throttle"

for heatmap_type in heatmap_type_list:

    data_csv = f"/home/jiaqq/Documents/ThirdEye-II/evaluate/roadGen/{heatmap_type}_{focus}_thirdeye.csv"
    df = pd.read_csv(data_csv)

    ttms = [1, 2, 3]
    summarization_method_list = ['average', 'average_gradient']
    aggregation_method_list = ["mean", "max"]

    results = []

    for summarization_method in summarization_method_list:
        for aggregation_method in aggregation_method_list:
            for ttm in ttms:

                filtered_df = df[
                    (df['summarization_method'] == summarization_method) &
                    (df['aggregation_method'] == aggregation_method) &
                    (df['ttm'] == ttm)
                    ]

                precision = round(filtered_df['precision'].mean())
                recall = round(filtered_df['recall'].mean())
                f3 = round(filtered_df['f3'].mean())

                results.append([focus, heatmap_type,
                    summarization_method, aggregation_method, ttm,
                    precision, recall, f3])

    output_csv = f"/home/jiaqq/Documents/ThirdEye-II/evaluate/roadGen/results.csv"
    results_df = pd.DataFrame(results, columns=['focus', 'heatmap_method', 'summarization_method',
                                                'aggregation_method', 'ttm', 'precision', 'recall', 'f3'])

    file_exists = os.path.exists(output_csv)
    results_df.to_csv(output_csv, mode='a', index=False, header=not file_exists)


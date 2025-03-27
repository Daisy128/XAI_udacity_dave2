import os.path
import pathlib

import pandas as pd

heatmap_type_list = ["smooth_grad", "raw_smooth_grad",
                     "grad_cam_pp", "raw_grad_cam_pp",
                     "faster_score_cam", "raw_faster_score_cam",
                     "integrated_gradients", "raw_integrated_gradients", ]
# heatmap_type_list = ['grad_cam_pp', 'raw_grad_cam_pp']
focus = "throttle"
track = "mountain"
root_dir = pathlib.Path("/home/jiaqq/Documents/ThirdEye-II/evaluate/final_evaluation")
attention = "seye"
# For seye: 'total_road_attention_ratio', "avg_road_attention_ratio"; for thirdeye: 'average', 'average_gradient'

for heatmap_type in heatmap_type_list:

    data_csv = os.path.join(root_dir, f"{heatmap_type}_{focus}_{attention}.csv")

    ttms = [1, 2, 3]
    aggregation_method_list = ["mean", "max"]
    if attention == "thirdeye":
        summarization_method_list = ['average', 'average_gradient'] # 'entropy', 'gini', 'topk', 'average', 'average_gradient']
    elif attention == "seye":
        summarization_method_list = ['total_road_attention_ratio', "avg_road_attention_ratio"]
    else:
        raise ValueError(f"Invalid attention type: {attention}")

    results = []

    for summarization_method in summarization_method_list:
        for aggregation_method in aggregation_method_list:
            for ttm in ttms:
                print(data_csv)
                df = pd.read_csv(data_csv)
                # df = df[(df['summarization_method'] == 'average_gradient') & (df['aggregation_method'] == 'max') & (df['ttm'] == 3)]

                df = df[(df['perturbation_folder'].str.startswith(track)) & (df['summarization_method'] == summarization_method)
                        & (df['aggregation_method'] == aggregation_method) & (df['ttm'] == ttm)]

                precision = round(df['precision'].mean())
                recall = round(df['recall'].mean())
                f3 = round(df['f3'].mean())

                results.append([focus, track, heatmap_type, summarization_method, aggregation_method, ttm, precision, recall, f3])


    output_csv = os.path.join(root_dir, "results", f"{track}_{focus}_{attention}.csv")
    results_df = pd.DataFrame(results, columns=['focus', 'track', 'heatmap_method', 'summarization_method',
                                                'aggregation_method', 'ttm', 'precision', 'recall', 'f3'])

    file_exists = os.path.exists(output_csv)
    results_df.to_csv(output_csv, mode='a', index=False, header=not file_exists)
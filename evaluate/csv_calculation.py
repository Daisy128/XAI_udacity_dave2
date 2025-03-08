import pandas as pd

heatmap_type_list = ["smooth_grad", "raw_smooth_grad",
                     "grad_cam_pp", "raw_grad_cam_pp",
                     "faster_score_cam", "raw_faster_score_cam",
                     "integrated_gradients", "raw_integrated_gradients", ]
focus = "throttle"

for heatmap_type in heatmap_type_list:

    data_csv = f"/home/jiaqq/Documents/ThirdEye-II/evaluate/{heatmap_type}_{focus}_seye.csv"

    ttms = [1, 2, 3]
    summarization_method_list = ["avg_road_attention_ratio"]
    aggregation_method_list = ["mean", "max"]

    results = []

    for summarization_method in summarization_method_list:
        for aggregation_method in aggregation_method_list:
            for ttm in ttms:
                print(data_csv)
                df = pd.read_csv(data_csv)
                # df = df[(df['summarization_method'] == 'average_gradient') & (df['aggregation_method'] == 'max') & (df['ttm'] == 3)]
                df = df[(df['summarization_method'] == summarization_method) & (df['aggregation_method'] == aggregation_method) & (df['ttm'] == ttm)]

                precision = round(df['precision'].mean())
                recall = round(df['recall'].mean())
                f3 = round(df['f3'].mean())

                results.append([focus, heatmap_type, summarization_method, aggregation_method, ttm, precision, recall, f3])

    results_df = pd.DataFrame(results, columns=['focus', 'heatmap_method', 'summarization_method', 'aggregation_method', 'ttm', 'precision', 'recall', 'f3'])
    # print(results_df.to_string())
    output_csv = f"/home/jiaqq/Documents/ThirdEye-II/evaluate/results.csv"
    results_df.to_csv(output_csv, mode='a', index=False)

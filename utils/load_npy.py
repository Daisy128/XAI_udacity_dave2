import pathlib

import numpy as np
import pandas as pd

if __name__ == '__main__':

    npy_file = pathlib.Path("/data/ThirdEye-II/perturbationdrive/logs/lake/lake_white_balance_filter_scale4_log/raw_grad_cam_pp_steer/"
                   "raw_grad_cam_pp_average_scores.npy")
    csv_file = pathlib.Path("/data/ThirdEye-II/perturbationdrive/logs/lake/lake_white_balance_filter_scale4_log/lake_white_balance_filter_scale4_log.csv")

    data_df = pd.read_csv(csv_file)

    data = np.load(npy_file)

    data_df["hm_score"] = data
    # data_df["hm_score_max"] = data_df["hm_score"].astype(np.float32)

    # print("Max value:", data[938].max())
    # print("Min value:", data[938].min())
    # print(print(np.isnan(data).sum()) )
    pd.DataFrame(data_df).to_csv("/home/jiaqq/Desktop/output.csv")

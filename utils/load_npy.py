import pathlib

import numpy as np
import pandas as pd

if __name__ == '__main__':

    npy_file = pathlib.Path("/home/jiaqq/Documents/ThirdEye-II/perturbationdrive/logs/lake/lake_glass_blur_scale7_log/smooth_grad_steer/"
                   "segment_total_road_attention_ratio.npy")

    data = np.load(npy_file)
    # data = data[~np.isnan(data)]
    data = data[data!=0]
    print("Data Shape:", data.shape)
    print("Data Type:", data.dtype)

    print(data)
    # print(print(np.isnan(data).sum()) )
    # pd.DataFrame(data).to_csv("/home/jiaqq/Desktop/output.csv")

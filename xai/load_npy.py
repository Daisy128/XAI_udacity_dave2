import numpy as np
import pandas as pd

data = np.load("/home/jiaqq/Documents/ThirdEye-II/perturbationdrive/logs/lake/lake_rotate_image_scale3_log/saliency_heatmap_steering/average_gradient_scores.npy")

print("Data Shape:", data.shape)
print("Data Type:", data.dtype)

print(data)
pd.DataFrame(data).to_csv("/home/jiaqq/Desktop/output.csv")

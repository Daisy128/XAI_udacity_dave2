import os

import numpy as np
from scipy.stats import entropy, kurtosis, gamma
from sklearn.preprocessing import normalize

def calculate_entropy(heatmap):
    prob = normalize(np.abs(heatmap).reshape(1, -1), norm='l1')[0]
    return entropy(prob, base=2)

def calculate_gini(heatmap):
    x = np.abs(heatmap.flatten())
    x = np.sort(x)
    n = len(x)
    return (np.sum((2 * np.arange(1, n+1) - n - 1) * x)) / (n * np.sum(x))

def top_k_percentage(heatmap, k=10):
    x = np.abs(heatmap.flatten())
    threshold = np.percentile(x, 100 - k)
    top_values = x[x >= threshold]
    return np.sum(top_values) / np.sum(x)

def get_gamma_threshold(losses, conf_level=0.95):
    losses = np.array(losses)
    losses_copy = losses[losses != 0]
    shape, loc, scale = gamma.fit(losses_copy, floc=0)
    t = gamma.ppf(conf_level, shape, loc=loc, scale=scale)
    return t

def get_percentile_threshold (losses, conf_level=0.95):
    threshold = np.percentile(losses, conf_level * 100)
    return threshold



if __name__ == '__main__':
    heatmap_type_list = ["smooth_grad", "raw_smooth_grad",
                         "raw_grad_cam_pp", "grad_cam_pp",
                         "faster_score_cam", "raw_faster_score_cam",
                         "integrated_gradients", "raw_integrated_gradients"]
    focus = "steer"

    root_dir = "/data/ThirdEye-II/perturbationdrive/logs/lake/"
    for heatmap_type in heatmap_type_list:
        for perturb_folder in os.listdir(root_dir):
            perturb_path = os.path.join(root_dir, perturb_folder)

            heatmap_file = os.path.join(perturb_path, f"{heatmap_type}_{focus}", f"{heatmap_type}_score.npy")
            print(f"Generating entropy for: {perturb_path}/{heatmap_type}_{focus}")

            if os.path.isfile(heatmap_file):
                heatmaps = np.load(heatmap_file)  # shape = (1921, 1, 80, 160)

                # entropy_scores = [calculate_entropy(hm.squeeze()) for hm in heatmaps]
                gini_scores = [calculate_gini(hm.squeeze()) for hm in heatmaps]
                topk_scores = [top_k_percentage(hm.squeeze(), k=10) for hm in heatmaps]

                # entropy_numpy = os.path.join(perturb_path, f"{heatmap_type}_{focus}", f"{heatmap_type}_entropy_scores.npy")
                gini_numpy = os.path.join(perturb_path, f"{heatmap_type}_{focus}",
                                             f"{heatmap_type}_gini_scores.npy")
                topk_numpy = os.path.join(perturb_path, f"{heatmap_type}_{focus}",
                                             f"{heatmap_type}_topk_scores.npy")

                # np.save(entropy_numpy, np.array(entropy_scores))
                np.save(gini_numpy, np.array(gini_scores))
                np.save(topk_numpy, np.array(topk_scores))

    # entropy_scores = np.load("/data/ThirdEye-II/perturbationdrive/logs/lake/lake_white_balance_filter_scale4_log/smooth_grad_steer/smooth_grad_entropy_score.npy")
    # print(entropy_scores) # 熵降到8或更低说明attention集中，接近13.6说明平铺
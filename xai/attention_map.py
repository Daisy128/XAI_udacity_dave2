import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tf_keras_vis.saliency import Saliency
from tqdm import tqdm
from utils import utils
from utils.conf import *

def score_increase(output):
    return output[:, 0]

def score_decrease(output):
    return -1.0 * output[:, 0]

def score_maintain(output):
    return tf.math.abs(1.0 / (output[:, 0] + tf.keras.backend.epsilon()))

def compute_heatmap(model, image_dir, csv_filename):
    data_df = pd.read_csv(csv_filename)
    heatmap_df = pd.DataFrame()
    heatmap_df[["index", "is_crashed", "origin_image_path"]] = data_df[["index", "is_crashed", "image_path"]].copy()

    saliency = Saliency(model, model_modifier=None)
    heatmap_dir = os.path.join(image_dir, "saliency_heatmap")
    os.makedirs(heatmap_dir, exist_ok=True)
    overlay_dir = os.path.join(image_dir, "saliency_heatmap_overlay")
    os.makedirs(overlay_dir, exist_ok=True)

    avg_heatmaps = []
    avg_gradient_heatmaps = []
    list_of_image_paths = []
    prev_hm = np.zeros((80, 160))

    for idx, img in tqdm(zip(heatmap_df["index"], heatmap_df["origin_image_path"]), total=len(heatmap_df)):

        x = np.asarray(Image.open(img))
        x = utils.resize(x).astype('float32')
        # print("image max value is: ", np.max(x)) # 231 ?
        saliency_map = saliency(score_decrease, x, smooth_samples=20, smooth_noise=0.20)
        gradient = abs(prev_hm - saliency_map) if idx != 1 else 0

        average = np.average(saliency_map)
        average_gradient = np.average(gradient)
        prev_hm = saliency_map

        avg_heatmaps.append(average)
        avg_gradient_heatmaps.append(average_gradient)

        saliency_map_path = os.path.join(heatmap_dir, f"heatmap_{idx}.png")
        plt.imsave(saliency_map_path, np.squeeze(saliency_map))

        list_of_image_paths.append(saliency_map_path)

        saliency_map_colored = plt.cm.viridis(np.squeeze(saliency_map))[:, :, :3]
        saliency_map_colored = (saliency_map_colored * 255).astype(np.uint8)
        overlay = (x * 0.5 + saliency_map_colored * 0.5).astype(np.uint8)
        overlay_path = os.path.join(overlay_dir, f"overlay_{idx}.png")
        plt.imsave(overlay_path, overlay)

    # save into files
    save_path = os.path.join(heatmap_dir, "average_scores.npy")
    np.save(save_path, avg_heatmaps) # saved as numpy arrays
    save_path = os.path.join(heatmap_dir, "average_gradient_scores.npy")
    np.save(save_path, avg_gradient_heatmaps)

    plt.hist(avg_heatmaps)
    plt.title("average attention heatmaps")
    save_path = os.path.join(heatmap_dir, "average_scores_hist.png")
    plt.savefig(save_path)
    plt.show()

    plt.hist(avg_gradient_heatmaps)
    plt.title("average gradient attention heatmaps")
    save_path = os.path.join(heatmap_dir, "average_gradient_scores_hist.png")
    plt.savefig(save_path)
    plt.show()

    heatmap_df['heatmap_image_path'] = list_of_image_paths

    heatmap_df.to_csv(os.path.join(heatmap_dir,'heatmap_log.csv'), index=True)

if __name__ == '__main__':
    track_index = 1 # lake, mountain or roadGenerator

    model = load_model(track_infos[track_index]["model_path"])

    root_dir = f"perturbationdrive/logs/{track_infos[track_index]['track_name']}"
    image_folder_name = "lake_defocus_blur_scale6_log"
    image_dir = os.path.join(root_dir,image_folder_name)

    csv_filename = os.path.join(image_dir, f"{image_folder_name}.csv")

    compute_heatmap(model, image_dir, csv_filename)
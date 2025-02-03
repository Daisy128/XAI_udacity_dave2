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

def score_increase(output, focus="steering"):
    if focus == "steering":
        return output[:, 0]
    elif focus == "throttle":
        return output[:, 1]
    elif focus == "both":
        return 0.5 * output[:, 0] + 0.5 * output[:, 1]
    else:
        raise ValueError("Invalid mode. Choose 'steering', 'throttle', or 'both'")

def score_decrease(output, focus="steering"):
    if focus == "steering":
        return -1.0 * output[:, 0]
    elif focus == "throttle":
        return -1.0 * output[:, 1]
    elif focus == "both":
        return -1.0 * (0.5 * output[:, 0] + 0.5 * output[:, 1])
    else:
        raise ValueError("Invalid mode. Choose 'steering', 'throttle', or 'both'")

def score_maintain(output, focus="steering"):
    if focus == "steering":
        return tf.math.abs(1.0 / (output[:, 0] + tf.keras.backend.epsilon()))
    elif focus == "throttle":
        return tf.math.abs(1.0 / (output[:, 1] + tf.keras.backend.epsilon()))
    elif focus == "both":
        return tf.math.abs(1.0 / ((0.5 * output[:, 0] + 0.5 * output[:, 1]) + tf.keras.backend.epsilon()))
    else:
        raise ValueError("Invalid mode. Choose 'steering', 'throttle', or 'both'")


def compute_heatmap(model, folder_path, csv_filename, focus):
    data_df = pd.read_csv(csv_filename)
    heatmap_df = pd.DataFrame()
    heatmap_df[["index", "is_crashed", "origin_image_path"]] = data_df[["index", "is_crashed", "image_path"]].copy()

    saliency = Saliency(model, model_modifier=None)
    heatmap_dir = os.path.join(folder_path, f"saliency_heatmap_{focus}")
    os.makedirs(heatmap_dir, exist_ok=True)
    overlay_dir = os.path.join(folder_path, f"saliency_heatmap_overlay_{focus}")
    os.makedirs(overlay_dir, exist_ok=True)

    avg_heatmaps = []
    avg_gradient_heatmaps = []
    list_of_image_paths = []
    prev_hm = np.zeros((80, 160))

    for idx, img in tqdm(zip(heatmap_df["index"], heatmap_df["origin_image_path"]), total=len(heatmap_df)):

        # for RoadGenerator, no need to resize
        x = np.asarray(Image.open(img), dtype=np.float32)

        #       for tracks model:
        #         x = np.asarray(Image.open(img))
        #         x = utils.resize(x).astype('float32')

        saliency_map = saliency(lambda output:score_decrease(output, focus), x, smooth_samples=20, smooth_noise=0.20)
        gradient = abs(prev_hm - saliency_map) if idx != 1 else 0

        average = np.average(saliency_map)
        average_gradient = np.average(gradient)
        prev_hm = saliency_map

        avg_heatmaps.append(average)
        avg_gradient_heatmaps.append(average_gradient)

        saliency_map_path = os.path.join(heatmap_dir, f"heatmap_{idx}.png")
        plt.imsave(saliency_map_path, np.squeeze(saliency_map))

        list_of_image_paths.append(saliency_map_path)

        saliency_map_colored = plt.cm.jet(np.squeeze(saliency_map))[:, :, :3]
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

def run_heatmap_tracks(track_index, focus):
    model_name = load_model(track_infos[track_index]["model_path"])
    root_folder = f"perturbationdrive/logs/{track_infos[track_index]['track_name']}"

    for folder_name in os.listdir(root_folder):
        print("Generating attention map on folder: ", folder_name)

        folder_path = os.path.join(root_folder, folder_name)
        # heatmap_dir = os.path.join(folder_path, f"saliency_heatmap_{focus}")

        if os.path.isdir(folder_path) and model_name is not None:
            csv_filename = os.path.join(folder_path, f"{folder_name}.csv")
            compute_heatmap(model_name, folder_path, csv_filename, focus)

def run_heatmap_roadGen(focus):
    model_name = load_model(roadGen_infos["model_path"])
    root_folder = f"perturbationdrive/logs/{roadGen_infos['track_name']}"

    for folder_name in sorted(os.listdir(root_folder)):
        parent_dir = os.path.join(root_folder, folder_name)

        for scaled_folder in sorted(os.listdir(parent_dir)):
            folder_path = os.path.join(parent_dir, scaled_folder)
            print("Generating attention map on folder: ", scaled_folder)

            if os.path.isdir(folder_path) and model_name is not None:
                csv_filename = os.path.join(folder_path, f"{scaled_folder}.csv")
                compute_heatmap(model_name, folder_path, csv_filename, focus)

if __name__ == '__main__':
    obj = "roadGen"
    track_index = 3 # lake == 1, mountain == 3
    focus = "throttle" # steering, or throttle

    if obj == "tracks":
        run_heatmap_tracks(track_index, focus)
    elif obj == "roadGen":
        run_heatmap_roadGen(focus)
    else:
        raise ValueError("Invalid object. Choose 'tracks' or 'roadGen'")
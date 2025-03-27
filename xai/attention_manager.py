import gc
import os
import pathlib
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from concurrent.futures import ThreadPoolExecutor
from keras import backend as K
from tensorflow.keras.models import load_model

from utils.conf import roadGen_infos
from utils.utils import preprocess, normalize, resize
from xai.attention_generator import AttentionMapGenerator


def save_overlay_image(image_resize, heatmap):
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)  # dpi: high-resolution output
    ax.imshow(image_resize)
    h, w, _ = np.array(image_resize).shape
    resized_heatmap = cv2.resize(heatmap, (w, h))
    ax.imshow(resized_heatmap, cmap='jet', alpha=0.4)
    ax.axis('off')

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除 padding
    plt.close(fig)

    return fig


class AttentionMapManager:

    def __init__(self, heatmap_config: dict, heatmap_generator: AttentionMapGenerator):
        self.args = heatmap_config['args']
        self.heatmap_function = heatmap_config['heatmap_function']
        self.save_images = heatmap_config['save_images']
        self.heatmap_generator = heatmap_generator

        if self.args['track_index'] == 1:
            self.args['track_name'] = "lake"
        elif self.args['track_index'] == 3:
            self.args['track_name'] = "mountain"
        else:
            raise ValueError("Invalid track index")

    def compute_heatmap(self, folder_path, csv_filename):

        data_df = pd.read_csv(csv_filename, usecols=["index", "frameId", "is_crashed", "image_path", "steer", "throttle"])

        heatmap_dir = os.path.join(folder_path, f"{self.args['function_name']}_{self.args['focus']}")
        os.makedirs(heatmap_dir, exist_ok=True)
        if self.save_images:
            overlay_dir = os.path.join(folder_path, f"{self.args['function_name']}_overlay_{self.args['focus']}")
            os.makedirs(overlay_dir, exist_ok=True)

        total_score = []
        predicts = []
        avg_heatmaps = []
        avg_gradient_heatmaps = []
        list_of_image_paths = []
        prev_hm = np.zeros((80, 160), dtype=np.float32)

        for frameId, img_path in tqdm(zip(data_df["frameId"], data_df["image_path"]), total=len(data_df)):
            # image preprocess
            img = Image.open(img_path)
            if self.args["obj"] == "tracks":
                image_resize, image= preprocess(img)
            elif self.args["obj"] == "roadGen":
                image = np.array(img, dtype=np.float32)
            else:
                raise ValueError("Invalid obj")

            # Run heatmap generation
            score, prediction = self.heatmap_function(image)
            total_score.append(score)
            predicts.append(prediction)

            # print("Score has shape: ", score.shape)
            gradient = abs(prev_hm - score) if frameId != 1 else 0
            average = np.average(score)
            average_gradient = np.average(gradient)
            prev_hm = score

            avg_heatmaps.append(average)
            avg_gradient_heatmaps.append(average_gradient)

            if self.save_images:
                heatmap = np.squeeze(score) # (1, 80, 160) -> (80, 160)
                # Debug:
                # print(f"Attribution min: {np.min(heatmap)}, max: {np.max(heatmap)}")
                heatmap_path = os.path.join(heatmap_dir, f"heatmap_{frameId}.png")
                plt.imsave(heatmap_path, heatmap, cmap='jet')
                plt.close()
                list_of_image_paths.append(heatmap_path)

                fig = save_overlay_image(image_resize, heatmap)
                fig.savefig(os.path.join(overlay_dir, f"overlay_{frameId}.png"))

        # saved as numpy arrays
        np.save(os.path.join(heatmap_dir, f"{self.args['function_name']}_score.npy"), total_score)
        np.save(os.path.join(heatmap_dir, f"{self.args['function_name']}_average_scores.npy"), avg_heatmaps)
        np.save(os.path.join(heatmap_dir, f"{self.args['function_name']}_average_gradient_scores.npy"), avg_gradient_heatmaps)

        plt.figure()
        # avg_heatmaps = np.nan_to_num(avg_heatmaps, nan=0)  # Replace NaN with 0
        plt.hist(avg_heatmaps)
        plt.title("average attention heatmaps")
        plt.savefig(os.path.join(heatmap_dir, "average_scores_hist.png"))
        plt.close()

        plt.figure()
        # avg_gradient_heatmaps = np.nan_to_num(avg_gradient_heatmaps, nan=0)  # Replace NaN with 0
        plt.hist(avg_gradient_heatmaps)
        plt.title("average gradient attention heatmaps")
        plt.savefig(os.path.join(heatmap_dir, "average_gradient_scores_hist.png"))
        plt.close()

        data_df[f'predicted_{self.args["focus"]}'] = predicts
        if self.save_images:
            data_df['heatmap_image_path'] = list_of_image_paths

        data_df.to_csv(os.path.join(heatmap_dir, 'heatmap_log.csv'), index=False)

        del avg_heatmaps, avg_gradient_heatmaps, list_of_image_paths, prev_hm
        K.clear_session()
        gc.collect()


    def run_heatmap(self):
        if self.args["mutate"] and self.args["obj"] == "tracks":
            self.run_heatmap_on_mutation_tracks()
        elif self.args["obj"] == "tracks":
            self.run_heatmap_tracks()
        elif self.args["mutate"] and self.args["obj"] == "roadGen":
            self.run_heatmap_on_mutation_roadGen()
        elif self.args["obj"] == "roadGen":
            self.run_heatmap_roadGen()
        else:
            raise ValueError("Invalid object type. Choose 'tracks' or 'roadGen'")

    def run_heatmap_tracks(self):
        # for testing
        root_folder = f"perturbationdrive/logs/{self.args['track_name']}/lake"
        # root_folder = pathlib.Path(f"/data/ThirdEye-II/perturbationdrive/logs/{self.args['track_name']}")
        for folder_name in os.listdir(root_folder):
            heatmap_folder = os.path.join(root_folder, folder_name, f"{self.args['function_name']}_{self.args['focus']}")
            if not os.path.isdir(heatmap_folder):
                print("Generating attention map on folder: ", folder_name)

                # perturbationdrive/logs/lake/lake_cutout_filter_scale8_log
                folder_path = os.path.join(root_folder, folder_name)

                if os.path.isdir(folder_path):
                    csv_filename = os.path.join(folder_path, f"{folder_name}.csv")
                    print("Computing heatmap on model: ", csv_filename, " ...")
                    self.compute_heatmap(folder_path, csv_filename)
            else:
                print("Heatmap for folder: ", folder_name, " already exists. Skipping.")

    def run_heatmap_on_mutation_roadGen(self):
        root_folder = pathlib.Path(f"mutation/logs/RoadGenerator")

        # folder_name == add_weights_regularisation_l1_6_log
        for folder_name in sorted(os.listdir(root_folder)):
            print("Generating attention map on folder: ", folder_name)
            parent_dir = os.path.join(root_folder, folder_name)

            for scaled_folder in sorted(os.listdir(parent_dir)): # roadGen_cutout_filter_road0_scale0_log
                heatmap_folder = os.path.join(parent_dir, scaled_folder,
                                              f"{self.args['function_name']}_{self.args['focus']}")
                if not os.path.isdir(heatmap_folder):
                    folder_path = os.path.join(parent_dir, scaled_folder)
                    print("Generating ", self.args['function_name'], " map on folder: ", scaled_folder)

                    if os.path.isdir(folder_path):
                        csv_filename = os.path.join(folder_path, f"{scaled_folder}.csv")
                        if os.path.exists(csv_filename):
                            df = pd.read_csv(csv_filename)
                            model = load_model(df.at[1, "model"], compile=False)
                            self.heatmap_generator.model = model
                            print("Computing heatmap on model: ", df.at[1, "model"], " ...")
                            self.compute_heatmap(folder_path, csv_filename)
                else:
                    print("Heatmap for folder: ", scaled_folder, " already exists. Skipping.")


    def run_heatmap_roadGen(self):
        root_folder = f"perturbationdrive/logs/{roadGen_infos['track_name']}" # logs/RoadGenerator

        for folder_name in sorted(os.listdir(root_folder)): # cutout_filter
            print("Generating attention map on folder: ", folder_name)
            parent_dir = os.path.join(root_folder, folder_name)

            for scaled_folder in sorted(os.listdir(parent_dir)): # roadGen_cutout_filter_road0_scale0_log
                heatmap_folder = os.path.join(parent_dir, scaled_folder,
                                              f"{self.args['function_name']}_{self.args['focus']}")
                if not os.path.isdir(heatmap_folder):
                    folder_path = os.path.join(parent_dir, scaled_folder)
                    print("Generating attention map on folder: ", scaled_folder)

                    if os.path.isdir(folder_path):
                        csv_filename = os.path.join(folder_path, f"{scaled_folder}.csv")
                        self.compute_heatmap(folder_path, csv_filename)
                else:
                    print("Heatmap for folder: ", scaled_folder, " already exists. Skipping.")

    def run_heatmap_on_mutation_tracks(self):
        root_folder = pathlib.Path(f"mutation/logs/{self.args['track_name']}")

        # folder_name == lake_add_weights_regularisation_l1_6_log
        for folder_name in sorted(os.listdir(root_folder)):
            # mutation/logs/lake/lake_add_weights_regularisation_l1_6_log
            folder_path = os.path.join(root_folder, folder_name)
            # mutation/logs/lake/lake_add_weights_regularisation_l1_6_log/smooth_grad_steer
            heatmap_folder = os.path.join(folder_path, f"{self.args['function_name']}_{self.args['focus']}")

            if not os.path.isdir(heatmap_folder):
                print("Generating attention map on folder: ", folder_name)
                # mutation/logs/lake/lake_add_weights_regularisation_l1_6_log/*.csv
                csv_filename = os.path.join(folder_path, f"{folder_name}.csv")
                if os.path.exists(csv_filename):
                    df = pd.read_csv(csv_filename)
                    model = load_model(df.at[1, "model"], compile=False)
                    # 在此处更新generate heatmap时会用到的model
                    self.heatmap_generator.model = model
                    print("Computing heatmap on model: ", df.at[1, "model"], " ...")
                    self.compute_heatmap(folder_path, csv_filename)

            else:
                print(self.args['function_name'], "for folder:", folder_name, "already exists. Skipping.")

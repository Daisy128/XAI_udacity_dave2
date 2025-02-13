import gc
import os
import pathlib
from tqdm import tqdm
from utils import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from utils.conf import roadGen_infos
from xai.attention_generator import AttentionMapGenerator


class AttentionMapManager:

    def __init__(self, heatmap_function, args: dict):
        self.args = args
        self.heatmap_function = heatmap_function


    def compute_heatmap(self, folder_path, csv_filename):

        data_df = pd.read_csv(csv_filename, usecols=["index", "is_crashed", "image_path"])

        heatmap_dir = os.path.join(folder_path, f"{self.args['function_name']}_{self.args['focus']}")
        os.makedirs(heatmap_dir, exist_ok=True)
        overlay_dir = os.path.join(folder_path, f"{self.args['function_name']}_overlay_{self.args['focus']}")
        os.makedirs(overlay_dir, exist_ok=True)

        total_score = []
        avg_heatmaps = []
        avg_gradient_heatmaps = []
        list_of_image_paths = []
        prev_hm = np.zeros((80, 160), dtype=np.float32)

        for idx, img in tqdm(zip(data_df["index"], data_df["image_path"]), total=len(data_df)):

            x = np.asarray(Image.open(img), dtype=np.float32)
            # for Tracks need to resize
            if self.args['track_name'] != "roadGen":
                x = utils.resize(x)

            score = self.heatmap_function(x)
            total_score.append(score)

            gradient = abs(prev_hm - score) if idx != 1 else 0
            average = np.average(score)
            average_gradient = np.average(gradient)
            prev_hm = score

            avg_heatmaps.append(average)
            avg_gradient_heatmaps.append(average_gradient)

            heatmap = os.path.join(heatmap_dir, f"heatmap_{idx}.png")
            plt.imsave(heatmap, np.squeeze(score))

            list_of_image_paths.append(heatmap)

            heatmap = plt.cm.jet(np.squeeze(score))[:, :, :3]
            heatmap = (heatmap * 255).astype(np.uint8)
            overlay = (x * 0.5 + heatmap * 0.5).astype(np.uint8)
            overlay_path = os.path.join(overlay_dir, f"overlay_{idx}.png")
            plt.imsave(overlay_path, overlay)

        # saved as numpy arrays
        np.save(os.path.join(heatmap_dir, f"{self.args['function_name']}_score.npy"), total_score)
        np.save(os.path.join(heatmap_dir, f"{self.args['function_name']}_average_scores.npy"), avg_heatmaps)
        np.save(os.path.join(heatmap_dir, f"{self.args['function_name']}_average_gradient_scores.npy"), avg_gradient_heatmaps)

        plt.figure()
        plt.hist(avg_heatmaps)
        plt.title("average attention heatmaps")
        plt.savefig(os.path.join(heatmap_dir, "average_scores_hist.png"))
        plt.clf()

        plt.figure()
        plt.hist(avg_gradient_heatmaps)
        plt.title("average gradient attention heatmaps")
        plt.savefig(os.path.join(heatmap_dir, "average_gradient_scores_hist.png"))
        plt.clf()

        data_df['heatmap_image_path'] = list_of_image_paths
        data_df.to_csv(os.path.join(heatmap_dir, 'heatmap_log.csv'), index=False)

        del avg_heatmaps, avg_gradient_heatmaps, list_of_image_paths, prev_hm
        gc.collect()


    def run_heatmap(self):
        if self.args["mutate"] and self.args["obj"] == "tracks":
            self.run_heatmap_on_mutation_tracks()
        elif self.args["obj"] == "tracks":
            self.run_heatmap_tracks()
        elif self.args["obj"] == "roadGen":
            self.run_heatmap_roadGen()
        else:
            raise ValueError("Invalid object type. Choose 'tracks' or 'roadGen'")

    def run_heatmap_tracks(self):
        root_folder = f"perturbationdrive/logs/{self.args['track_name']}"

        for folder_name in os.listdir(root_folder):
            print("Generating attention map on folder: ", folder_name)

            # perturbationdrive/logs/lake/lake_cutout_filter_scale8_log
            folder_path = os.path.join(root_folder, folder_name)

            if os.path.isdir(folder_path):
                csv_filename = os.path.join(folder_path, f"{folder_name}.csv")
                self.compute_heatmap(folder_path, csv_filename)

    def run_heatmap_roadGen(self):
        root_folder = f"perturbationdrive/logs/{roadGen_infos['track_name']}"

        for folder_name in sorted(os.listdir(root_folder)):
            print("Generating attention map on folder: ", folder_name)
            parent_dir = os.path.join(root_folder, folder_name)

            for scaled_folder in sorted(os.listdir(parent_dir)):
                folder_path = os.path.join(parent_dir, scaled_folder)
                print("Generating attention map on folder: ", scaled_folder)

                if os.path.isdir(folder_path):
                    csv_filename = os.path.join(folder_path, f"{scaled_folder}.csv")
                    self.compute_heatmap(folder_path, csv_filename)

    def run_heatmap_on_mutation_tracks(self):
        root_folder = pathlib.Path(f"mutation/logs/{self.args['track_name']}")

        # folder_name == lake_add_weights_regularisation_l1_6_log
        for folder_name in sorted(os.listdir(root_folder)):
            print("Generating attention map on folder: ", folder_name)
            # mutation/logs/lake/lake_add_weights_regularisation_l1_6_log
            folder_path = os.path.join(root_folder, folder_name)
            csv_filename = os.path.join(folder_path, f"{folder_name}.csv")
            if os.path.exists(csv_filename):
                df = pd.read_csv(csv_filename)
                model = load_model(df.at[1, "model"])
                print("Computing heatmap on model: ", model, " ...")

                self.compute_heatmap(folder_path, csv_filename)

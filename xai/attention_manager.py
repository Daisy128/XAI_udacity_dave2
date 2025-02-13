import os
import pathlib

from tqdm import tqdm
from utils import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model

from utils.conf import track_infos, roadGen_infos
from xai.attention_generator import AttentionMapGenerator


class AttentionMapManager:
    def __init__(self, obj, track_name, model, mutate, attention= "smooth_grad", focus="steering"):
        self.obj = obj
        self.track_name = track_name
        self.model = model
        self.mutate = mutate
        self.attention = attention
        self.focus = focus

        map_generator = AttentionMapGenerator(model=model, focus=focus)
        attention_methods = {
            "SmoothGrad": map_generator.smooth_grad,
            "RawSmoothGrad": map_generator.raw_smooth_grad,
            "GradCAM++": map_generator.grad_cam_pp,
            "Faster-ScoreCAM": map_generator.faster_score_cam,
            "IntegratedGradients": map_generator.integrated_gradients
        }
        self.generator = attention_methods.get(attention)

    def run_heatmap(self):
        if self.mutate and self.obj == "tracks":
            self.run_heatmap_on_mutation_tracks()
        elif self.obj == "tracks":
            self.run_heatmap_tracks()
        elif self.obj == "roadGen":
            self.run_heatmap_roadGen()
        else:
            raise ValueError("Invalid object type. Choose 'tracks' or 'roadGen'")

    def run_heatmap_tracks(self):
        root_folder = f"perturbationdrive/logs/{self.track_name}"

        for folder_name in os.listdir(root_folder):
            print("Generating attention map on folder: ", folder_name)

            # perturbationdrive/logs/lake/lake_cutout_filter_scale8_log
            folder_path = os.path.join(root_folder, folder_name)
            # heatmap_dir = os.path.join(folder_path, f"saliency_heatmap_{focus}")

            if os.path.isdir(folder_path):
                csv_filename = os.path.join(folder_path, f"{folder_name}.csv")
                self.compute_heatmap(folder_path, csv_filename, "Tracks")

    def run_heatmap_roadGen(self):
        model = load_model(roadGen_infos["model_path"])
        root_folder = f"perturbationdrive/logs/{roadGen_infos['track_name']}"

        for folder_name in sorted(os.listdir(root_folder)):
            print("Generating attention map on folder: ", folder_name)
            parent_dir = os.path.join(root_folder, folder_name)

            for scaled_folder in sorted(os.listdir(parent_dir)):
                folder_path = os.path.join(parent_dir, scaled_folder)
                print("Generating attention map on folder: ", scaled_folder)

                if os.path.isdir(folder_path):
                    csv_filename = os.path.join(folder_path, f"{scaled_folder}.csv")
                    self.compute_heatmap(folder_path, csv_filename, "RoadGenerator")

    def run_heatmap_on_mutation_tracks(self):
        root_folder = pathlib.Path(f"mutation/logs/{self.track_name}")

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

    def compute_heatmap(self, folder_path, csv_filename, benchmark: str="Tracks"):
        data_df = pd.read_csv(csv_filename)
        heatmap_df = pd.DataFrame()
        heatmap_df[["index", "is_crashed", "origin_image_path"]] = data_df[["index", "is_crashed", "image_path"]].copy()

        heatmap_dir = os.path.join(folder_path, f"{self.attention}_{self.focus}")
        os.makedirs(heatmap_dir, exist_ok=True)
        overlay_dir = os.path.join(folder_path, f"{self.attention}_overlay_{self.focus}")
        os.makedirs(overlay_dir, exist_ok=True)

        avg_heatmaps = []
        avg_gradient_heatmaps = []
        list_of_image_paths = []
        prev_hm = np.zeros((80, 160))

        for idx, img in tqdm(zip(heatmap_df["index"], heatmap_df["origin_image_path"]), total=len(heatmap_df)):

            # for RoadGenerator, no need to resize
            if benchmark == "RoadGenerator":
                x = np.asarray(Image.open(img), dtype=np.float32)
            elif benchmark == "Tracks":
                x = np.asarray(Image.open(img))
                x = utils.resize(x).astype('float32')
            else:
                raise ValueError("Invalid benchmark. Choose 'Tracks' or 'RoadGenerator'")

            score = self.generator(x)

            gradient = abs(prev_hm - score) if idx != 1 else 0
            average = np.average(score)
            average_gradient = np.average(gradient)
            prev_hm = score

            avg_heatmaps.append(average)
            avg_gradient_heatmaps.append(average_gradient)

            saliency_map_path = os.path.join(heatmap_dir, f"heatmap_{idx}.png")
            plt.imsave(saliency_map_path, np.squeeze(score))

            list_of_image_paths.append(saliency_map_path)

            saliency_map_colored = plt.cm.jet(np.squeeze(score))[:, :, :3]
            saliency_map_colored = (saliency_map_colored * 255).astype(np.uint8)
            overlay = (x * 0.5 + saliency_map_colored * 0.5).astype(np.uint8)
            overlay_path = os.path.join(overlay_dir, f"overlay_{idx}.png")
            plt.imsave(overlay_path, overlay)

        # save into files
        save_path = os.path.join(heatmap_dir, f"{self.attention}_score.npy")
        np.save(save_path, score)  # saved as numpy arrays
        save_path = os.path.join(heatmap_dir, f"{self.attention}_average_scores.npy")
        np.save(save_path, avg_heatmaps)
        save_path = os.path.join(heatmap_dir, f"{self.attention}_average_gradient_scores.npy")
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
        heatmap_df.to_csv(os.path.join(heatmap_dir, 'heatmap_log.csv'), index=True)


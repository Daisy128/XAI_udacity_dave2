import gc
import os
import pathlib

from tqdm import tqdm

from utils.utils import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from utils.conf import roadGen_infos


class AttentionMapManager:

    def __init__(self, heatmap_config: dict):
        self.args = heatmap_config['args']
        self.heatmap_function = heatmap_config['heatmap_function']
        self.save_images = heatmap_config['save_images']

        if self.args['track_index'] == 1:
            self.args['track_name'] = "lake"
        elif self.args['track_index'] == 3:
            self.args['track_name'] = "mountain"
        else:
            raise ValueError("Invalid track index")

    def save_overlay_image(self, original_img, heatmap):
        fig, ax = plt.subplots(figsize=(20, 10), dpi=100)  # dpi: high-resolution output
        ax.imshow(original_img)
        ax.imshow(heatmap, cmap='jet', alpha=0.4)
        ax.axis('off')
        # # 关闭坐标轴和边界
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_frame_on(False)

        # 直接匹配画布和图像尺寸
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除 padding
        plt.close(fig)

        return fig


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

        for frameId, img in tqdm(zip(data_df["frameId"], data_df["image_path"]), total=len(data_df)):
            # image preprocess
            image = np.array(Image.open(img))
            image_crop = crop(image)
            image_resize = resize(image_crop)
            image_yuv = rgb2yuv(image_resize)
            image_nor = normalize(image_yuv)

            score, prediction = self.heatmap_function(image_nor)
            total_score.append(score)
            predicts.append(prediction)

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

                fig = self.save_overlay_image(image_resize, heatmap)
                fig.savefig(os.path.join(overlay_dir, f"overlay_{frameId}.png"))

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

        data_df[f'predicted_{self.args["focus"]}'] = predicts
        if self.save_images:
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
        # for testing
        # root_folder = f"perturbationdrive/logs/{self.args['track_name']}/lake"
        root_folder = f"perturbationdrive/logs/{self.args['track_name']}"
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
            # else:
            #     print("Heatmap for folder: ", folder_name, " already exists. Skipping.")

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

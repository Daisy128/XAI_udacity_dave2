import glob
import os
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.utils import crop, resize


def merge(saliency_map, predicted_rgb, debug=False):
    saliency_map = np.squeeze(saliency_map) # saliency_map.shape = (1, 80, 160) 3D

    # 浅色或白色  非道路区域
    non_road_mask = (predicted_rgb[:, :, 0] >= 100) & \
                    (predicted_rgb[:, :, 1] >= 100) & \
                    (predicted_rgb[:, :, 2] >= 100)

    saliency_map_seg = np.copy(saliency_map)  # 对 saliency_map 的修改不会影响原始数组
    saliency_map_seg[non_road_mask] = 0  # 只有道路区域的 attention 分数被保留

    # 向量级计算
    all_attention = np.sum(saliency_map) # sum of attention on the complete image
    all_pixel = saliency_map.size # sum of pixel on the complete image
    road_attention = np.sum(saliency_map_seg)  # 只有道路区域的 attention
    road_pixel = np.sum(~non_road_mask)  # 道路区域的像素数量

    avg_road_attention = road_attention / road_pixel if road_pixel > 0 else 0
    avg_all_attention = all_attention / all_pixel if all_pixel > 0 else 0

    if debug:
        plt.imshow(non_road_mask, cmap="gray")
        plt.title("Non-Road Mask (White = Non-Road)")
        plt.show()

        plt.imshow(saliency_map_seg, cmap="hot")
        plt.title("Filtered Saliency Map (Road Only)")
        plt.show()

    # for x in range(saliency_map.shape[0]):
    #     for y in range(saliency_map.shape[1]):
    #         all_attention += saliency_map_seg[x, y] # 整个图像的saliency_score总和
    #         ####################
    #         all_pixel += 1 # 整个图像像素数量
    #         ####################
    #         if np.all(predicted_rgb[x, y] >= [100, 100, 100]): # 浅色 & 白色
    #             saliency_map_seg[x, y] = 0 # 不在道路区域的像素saliency_score设为0
    #         else:
    #             road_pixel += 1 # 在道路区域的像素数量
    #             road_attention += saliency_map[x, y] # 在道路区域的saliency_score总和
    #
    # # In case of ads is crashed with no road part in img
    # if road_pixel == 0:
    #     avg_road_attention = 0
    # else:
    #     avg_road_attention = road_attention / road_pixel # 道路区域saliency_score总和 / 在道路区域的像素数量总和
    #
    # avg_all_attention = all_attention / all_pixel  # 整张图片的saliency_score总和 / 总像素数量

    return saliency_map_seg, avg_road_attention, avg_all_attention, road_attention, all_attention


if __name__ == '__main__':

    heatmap_type_list = ["smooth_grad","raw_smooth_grad",
                         "grad_cam_pp", "raw_grad_cam_pp",
                         "faster_score_cam", "raw_faster_score_cam",
                         "integrated_gradients", "raw_integrated_gradients", ]
    # heatmap_type_list = ["grad_cam_pp"]#, "raw_grad_cam_pp"]

    focus_list = ["steer", "throttle"]
    track = "mountain"
    focus = "throttle"
    root_path = pathlib.Path("/home/jiaqq/Documents/ThirdEye-II")  # "/home/jiaqq/Documents/ThirdEye-II" or "/data/ThirdEye-II"
    image_root = root_path/'mutation'/'logs'/track  # TODO: Apply dynamic path for other .py

    for folder_name in os.listdir(image_root):
        for heatmap_type in heatmap_type_list:
            print("Start generating seg_hm for", folder_name, "using", heatmap_type)
            # 读取 .npy 文件中所有图片的 heatmap_scores
            heatmap_dir = os.path.join(image_root, folder_name, f"{heatmap_type}_{focus}")

            if not glob.glob(os.path.join(heatmap_dir, "segment_*")):
                heatmap_scores = np.load(f"{heatmap_dir}/{heatmap_type}_score.npy")

                # 读取 csv 文件中每张图片的 frameId, 以对应其 segmentation
                data_csv = pd.read_csv(f"{image_root}/{folder_name}/{folder_name}.csv")

                list_of_total_road_attention_ratio = []
                list_of_avg_road_attention_ratio = []
                list_of_road_attention = []
                list_of_all_attention = []
                list_of_avg_road_attention = []
                list_of_avg_all_attention = []
                avg_gradient_heatmaps_seg=[]
                avg_heatmaps_seg=[]
                gradient_seg = prev_hm_seg = np.zeros((80, 160))

                # image with index "x" in data_csv corresponds to heatmap_scores[x+1]
                # iterate through all images with each one's segmented map and heatmap score
                for index, frameId in tqdm(zip(data_csv["index"], data_csv["frameId"]), total=len(data_csv)):
                    # print("Processing image: ", frameId)

                    # segmentation_map
                    seg_img = Image.open(f"{image_root}/{folder_name}/image_logs/computed_segmentation_{track}_sun/computed_{frameId}.png")
                    # crop and resize for segmentation image
                    seg_img_crop = crop(np.array(seg_img))
                    seg_img_resize = resize(seg_img_crop)

                    saliency_map_seg, avg_road_attention, avg_all_attention, road_attention, all_attention = merge(heatmap_scores[index-1], seg_img_resize)

                    if all_attention == 0:
                        print("all attention zero")
                    elif road_attention == 0:
                        print("road attention zero")
                    elif avg_all_attention == 0:
                        print("average attention zero")
                    elif avg_road_attention == 0:
                        print("average road attention zero")

                    list_of_total_road_attention_ratio.append(road_attention / all_attention)
                    list_of_avg_road_attention_ratio.append(avg_road_attention / avg_all_attention)
                    list_of_road_attention.append(road_attention)
                    list_of_all_attention.append(all_attention)
                    list_of_avg_road_attention.append(avg_road_attention)
                    list_of_avg_all_attention.append(avg_all_attention)

                    gradient_seg = abs(prev_hm_seg - saliency_map_seg) if index != 1 else 0
                    average_seg = np.average(saliency_map_seg)
                    average_gradient_seg = np.average(gradient_seg)
                    prev_hm_seg = saliency_map_seg

                    avg_heatmaps_seg.append(average_seg)
                    avg_gradient_heatmaps_seg.append(average_gradient_seg) # only average_gradient in Seye

                # np.save(os.path.join(heatmap_dir, f"segment_road_attention.npy"), list_of_road_attention)
                # np.save(os.path.join(heatmap_dir, f"segment_all_attention.npy"), list_of_all_attention)
                # np.save(os.path.join(heatmap_dir, f"segment_avg_road_attention.npy"), list_of_avg_road_attention)
                # np.save(os.path.join(heatmap_dir, f"segment_avg_all_attention.npy"), list_of_avg_all_attention)
                np.save(os.path.join(heatmap_dir, f"segment_avg_road_attention_ratio.npy"), list_of_avg_road_attention_ratio)
                np.save(os.path.join(heatmap_dir, f"segment_total_road_attention_ratio.npy"), list_of_total_road_attention_ratio)

                # np.save(os.path.join(heatmap_dir, f"segment_average.npy"), avg_heatmaps_seg)
                # np.save(os.path.join(heatmap_dir, f"segment_average_gradient.npy"), avg_gradient_heatmaps_seg)

                plt.figure()
                plt.hist(avg_heatmaps_seg)
                plt.title("average attention seg_heatmaps")
                plt.savefig(os.path.join(heatmap_dir, "segment_average.png"))
                plt.close()

                plt.figure()
                plt.hist(avg_gradient_heatmaps_seg)
                plt.title("average gradient attention seg_heatmaps")
                plt.savefig(os.path.join(heatmap_dir, "segment_average_gradient.png"))
                plt.close()












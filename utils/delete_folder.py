import glob
import pathlib
import shutil
import os

def delete_folder(folder_path):
    """
    删除指定的文件夹及其中的所有文件和子文件夹。

    参数:
    folder_path (str): 要删除的文件夹的路径。
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"文件夹 '{folder_path}' 已成功删除。")
    else:
        print(f"文件夹 '{folder_path}' 不存在。")

heatmap_list = ["smooth_grad_steer", "raw_smooth_grad_steer",
                "integrated_gradients_steer", "raw_integrated_gradients_steer",
                "grad_cam_pp_steer", "raw_grad_cam_pp_steer",
                "faster_score_cam_steer", "raw_faster_score_cam_steer"]

if __name__ == "__main__":
    root_path = pathlib.Path("/home/jiaqq/Documents/ThirdEye-II/perturbationdrive/logs/lake")
    for perturb_folder in os.listdir(root_path):
        perturb_path = os.path.join(root_path, perturb_folder)
        for folder_name in os.listdir(perturb_path):
            folder_path = os.path.join(perturb_path, folder_name)
            heatmap_files = glob.glob(os.path.join(folder_path, "segment_*"))

            for file in heatmap_files:
                os.remove(file)
            # folder_to_delete = os.path.join(folder_path, heatmap)
            # print(folder_to_delete)# delete_folder(folder_to_delete)
            # delete_folder(folder_to_delete)

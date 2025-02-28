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

heatmap_list = ["Faster-ScoreCAM_0", "Faster-ScoreCAM_overlay_0", "GradCAM++_0", "GradCAM++_overlay_0", "saliency_heatmap_overlay_steering", "saliency_heatmap_overlay_throttle", "saliency_heatmap_steering", "saliency_heatmap_throttle"]

if __name__ == "__main__":
    root_path = pathlib.Path("/home/jiaqq/Documents/ThirdEye-II/perturbationdrive/logs/lake")
    for heatmap in heatmap_list:
        for folder_name in os.listdir(root_path):
            folder_path = os.path.join(root_path, folder_name)
            folder_to_delete = os.path.join(folder_path, heatmap)
            print(folder_to_delete)# delete_folder(folder_to_delete)

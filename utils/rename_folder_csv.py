import csv
import pathlib
import shutil
import os


if __name__ == "__main__":
    # target: change original recorded data names
    method_names = {
        "SmoothGrad": "smooth_grad",
        "RawSmoothGrad": "raw_smooth_grad",
        "GradCAM++": "grad_cam_pp",
        "Faster-ScoreCAM": "faster_score_cam",
        "IntegratedGradients": "integrated_gradients"
    }


    # 设置目录路径
    root_directory = '/home/jiaqq/Documents/ThirdEye-II/perturbationdrive/logs/lake/'
    old_string = "IntegratedGradients"
    new_string = "integrated_gradients"
    to_rename = []
    csv_files = []

    for root, dirs, files in os.walk(root_directory, topdown=False):  # topdown=False确保从子目录开始
        # 修改文件名
        for file in files:
            if old_string in file:
                old_file_path = os.path.join(root, file)
                new_file_name = file.replace(old_string, new_string)
                new_file_path = os.path.join(root, new_file_name)
                to_rename.append((old_file_path, new_file_path))
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

        # 修改文件夹名
        for dir in dirs:
            if old_string in dir:
                old_dir_path = os.path.join(root, dir)
                new_dir_name = dir.replace(old_string, new_string)
                new_dir_path = os.path.join(root, new_dir_name)
                to_rename.append((old_dir_path, new_dir_path))

    # 执行重命名
    for old_path, new_path in to_rename:
        os.rename(old_path, new_path)
        print(f"Renamed '{old_path}' to '{new_path}'")

    for csv_file in csv_files:
        temp_file = csv_file + '.tmp'
        with open(csv_file, mode='r', newline='') as file, open(temp_file, mode='w', newline='') as outfile:
            reader = csv.reader(file)
            writer = csv.writer(outfile)
            for row in reader:
                new_row = [item.replace(old_string, new_string) for item in row]
                writer.writerow(new_row)
        os.replace(temp_file, csv_file)  # Replace the original file with the modified one

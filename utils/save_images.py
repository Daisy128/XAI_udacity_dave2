import os
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def save_image(image_path, image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(image_path)

def write_driving_log(csv_path, data):
    data_df = pd.DataFrame(data)
    data_df.index += 1
    data_df.to_csv(csv_path, index_label='index', encoding='utf-8', mode='w')

def save_data_in_batch(log_name, log_path, data, image_data):

    """
    Saves data and images in batches using a thread pool for efficient I/O operations and writes 
    associated metadata to a .csv log file.

    Args:
        log_name (str): Name of the csv log file to save the metadata.
        log_path (str): Directory path to store the csv log file and "image_logs" folder. The "image_logs" folder will be created if it doesn't exist.'
        data (list of dict): Metadata to be written in the csv. Each dictionary represents a row in the .csv file.
        image_data (list of tuple): A list of tuples where each tuple consists of:
            - image_path (str): The path where the image will be saved.
            - image (np.ndarray or PIL.Image): The image data to be saved.

    Raises:
        Exception: If any image saving operation fails, logs the error.
    """
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(save_image, image_path, image)
            for image_path, image in image_data
        ]
    # 等待所有任务完成
    for future in futures:
        try:
            future.result()  # 检查任务状态, 确保没有failed save
        except Exception as e:
            print(f"Error during image saving: {e}")
            # TODO: change to system-level logger
            
    write_driving_log(os.path.join(log_path, log_name), data)
    print(f"Data saved under {log_name}!")
    futures.clear()

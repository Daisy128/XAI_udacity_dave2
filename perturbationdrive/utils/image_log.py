import os
import csv
import cv2
import numpy as np
from PIL import Image


def perturb_driving_log(csv_path, data):
    with open(csv_path, 'w', newline='') as csvfile:
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"{csv_path} will be overwritten")


    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if data:
            writer.writerow(data[0].keys())  # column names
            for row in data:
                writer.writerow(row.values())

def save_image(image_path, image):

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image.save(image_path)
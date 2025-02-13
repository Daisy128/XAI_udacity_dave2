# to run it you have to:

# 1. pip install git+https://github.com/facebookresearch/segment-anything.git

# 2. download the sam_vit_h_4b8939.pth model from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints

#
import os
import cv2
import numpy as np

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch
import matplotlib.pyplot as plt

# Set up SAM model
import gc
gc.collect()
torch.cuda.empty_cache()

MODEL_PATH = os.path.join("model", "ckpts", "segmentation", "sam_vit_l_0b3195.pth") #"sam_vit_h_4b8939.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM = sam_model_registry["vit_l"](checkpoint=MODEL_PATH).to(DEVICE)
MASK_GENERATOR = SamAutomaticMaskGenerator(SAM)


# sam_checkpoint = os.path.join("..", "ckpts", "segmentation", "sam_vit_b_01ec64.pth") #"sam_vit_h_4b8939.pth"
# assert os.path.exists(sam_checkpoint)
# model_type = "vit_b"#"vit_h"
# DEVICE = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
# SAM = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(DEVICE)
# PREDICTOR = SamPredictor(SAM)


image_folder = os.path.join("Data", "lane_keeping_data", "test")

output_folder = os.path.join(image_folder, "masks")
os.makedirs(output_folder, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))]

if not image_files:
    print("No images found in the folder.")
    exit()


def compute_average_color(image, mask):
    masked_pixels = image[mask > 0]
    if len(masked_pixels) == 0:
        return (0, 0, 0)  # Default to black if no pixels are found
    return np.mean(masked_pixels, axis=0).astype(int)


def generate_masks_and_plot(image_files):

    for idx, image_file in enumerate(image_files):
        fig, axes = plt.subplots( 1,2, figsize=(10, 5))
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate masks using SAM
        masks = MASK_GENERATOR.generate(image_rgb)
        mask_image = np.zeros_like(image_rgb)

        for mask in masks:
            avg_color = compute_average_color(image_rgb, mask["segmentation"])
            mask_image[mask["segmentation"] > 0] = avg_color

        # Save mask visualization
        mask_path = os.path.join(output_folder, image_file.replace(".jpg", "_mask.png"))
        # cv2.imwrite(mask_path, cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR))

        # Plot images
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(mask_image)
        axes[1].set_title("Segmented Mask")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "combi_"+ image_file.replace(".jpg", ".png")))
        print(f"Mask saved to: {mask_path}")


# Process and plot all images
generate_masks_and_plot(image_files)

print("All images processed!")
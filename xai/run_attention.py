import os
from utils.conf import *
from tensorflow.keras.models import load_model
from attention_manager import AttentionMapManager
from xai.attention_generator import AttentionMapGenerator

data = {
    "object_name": "tracks",
    "track_name": "lake", # lake == 1, mountain == 3
    "mutate": False,
    "attention": "GradCAM++",
    "focus": "steering" # steering, or throttle
}
if __name__ == '__main__':
    model = load_model(track_infos[1]["model_path"])
    attention_manager = AttentionMapManager(obj=data["object_name"],
                                            track_name=data["track_name"],
                                            model=model,
                                            mutate=data["mutate"],
                                            attention=data["attention"],
                                            focus=data["focus"])


    attention_manager.run_heatmap()
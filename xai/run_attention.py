import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only warning 和 error

from utils.conf import *
from tensorflow.keras.models import load_model
from attention_manager import AttentionMapManager
from xai.attention_generator import AttentionMapGenerator

args = {
    "obj": "",
    "track_index": 1,  # 1 or 3
    "function_name": "",
    "focus": "",  # steer or throttle
    "mutate": False
}

heatmap_method_list = ["smooth_grad", "raw_smooth_grad",
                       "grad_cam_pp", "raw_grad_cam_pp",
                       "faster_score_cam", "raw_faster_score_cam",
                       "integrated_gradients", "raw_integrated_gradients", ]

if __name__ == '__main__':

    args["obj"] = "tracks"
    args["track_index"] = 3
    args["focus"] = "steer"
    args["mutate"] = False

    model = load_model(track_infos[args['track_index']]["model_path"])
    print("model loaded from: ", track_infos[args['track_index']]["model_path"])

    for heatmap_function in heatmap_method_list:
        print("Start generating heatmap: ", heatmap_function)

        args["function_name"] = heatmap_function
        heatmap_generator = AttentionMapGenerator(model=model, focus=args["focus"])

        heatmap_config = dict()
        heatmap_config["args"] = args
        heatmap_config["heatmap_function"] = getattr(heatmap_generator, args["function_name"])
        heatmap_config["save_images"] = False

        attention_manager = AttentionMapManager(heatmap_config = heatmap_config)

        attention_manager.run_heatmap()

        print("finish generating heatmap: ", heatmap_function)

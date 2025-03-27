import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only warning 和 error

from utils.conf import *
from tensorflow.keras.models import load_model
from xai.attention_manager import AttentionMapManager
from xai.attention_generator import AttentionMapGenerator

args = {
    "obj": "",
    "track_index": 1,  # 1 or 3
    "function_name": "",
    "focus": "",  # steer or throttle
    "mutate": False
}

heatmap_method_list = ["raw_grad_cam_pp", "grad_cam_pp",
                       "faster_score_cam", "raw_faster_score_cam",
                       "integrated_gradients", "raw_integrated_gradients",
                       "smooth_grad", "raw_smooth_grad",]

if __name__ == '__main__':

    args["obj"] = "roadGen" # "tracks" or "roadGen"
    args["track_index"] = 3 # if "roadGen" then not used
    args["focus"] = "steer"
    args["mutate"] = True # True for generating heatmaps on mutation and False for perturbation

    if args["obj"] == "tracks":
        model = load_model(track_infos[args['track_index']]["model_path"])
        # model = load_model("/home/jiaqq/Documents/ThirdEye-II/model/ckpts/ads-mutation/change_activation_function_exponential_2-6/track1_lake/track1-dave2-change_activation_function-20250315_152542-final.h5")
        print("model loaded from: ", track_infos[args['track_index']]["model_path"])
    elif args["obj"] == "roadGen":
        model = load_model(roadGen_infos['model_path'])
        print("model loaded from: ", roadGen_infos['model_path'])

    for heatmap_function in heatmap_method_list:
        print("Start generating heatmap: ", heatmap_function)

        args["function_name"] = heatmap_function
        heatmap_generator = AttentionMapGenerator(model=model, focus=args["focus"])

        heatmap_config = dict()
        heatmap_config["args"] = args
        heatmap_config["heatmap_function"] = getattr(heatmap_generator, args["function_name"])
        heatmap_config["save_images"] = False # True for generating visual images and overlay images, False for saving numpy scores only

        attention_manager = AttentionMapManager(heatmap_config = heatmap_config, heatmap_generator = heatmap_generator)

        attention_manager.run_heatmap()

        print("finish generating heatmap: ", heatmap_function)

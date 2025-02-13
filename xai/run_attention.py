from utils.conf import *
from tensorflow.keras.models import load_model
from attention_manager import AttentionMapManager
from xai.attention_generator import AttentionMapGenerator


heatmap_focus = {
    "steering": 0,
    "throttle": 1
}

method_names = {
    "SmoothGrad": "smooth_grad",
    "RawSmoothGrad": "raw_smooth_grad",
    "GradCAM++": "grad_cam_pp",
    "Faster-ScoreCAM": "faster_score_cam",
    "IntegratedGradients": "integrated_gradients"
}

if __name__ == '__main__':

    args = {
        "obj": "tracks",
        "track_id": 1,
        "track_name": "lake",
        "function_name": "GradCAM++",
        "mutate": False
    }
    focus = "steering"

    focus = heatmap_focus.get(focus)
    model = load_model(track_infos[args['track_id']]["model_path"])

    args["focus"] = focus
    heatmap_generator = AttentionMapGenerator(model=model, focus=focus)

    heatmap_function = dict()
    heatmap_function["args"] = args
    heatmap_function["function"] = getattr(heatmap_generator, method_names[args["function_name"]])

    attention_manager = AttentionMapManager(heatmap_function = heatmap_function.get("function"),
                                            args=heatmap_function.get("args"))

    attention_manager.run_heatmap()

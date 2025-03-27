import re
import sys

from evaluate.evaluate_failure_prediction_heatmaps_scores_roadGen import evaluate_failure_prediction_roadGen
from evaluate_failure_prediction_heatmaps_scores_percent import *
# from evaluate_failure_prediction_heatmaps_scores import *


def run_evaluation_roadGen(aggregation_method, summary_type, heatmap_type, focus):
  for perturb_type in os.listdir(log_dir):
    if perturb_type == "roadGen_normal":
        continue

    sub_log_path = os.path.join(log_dir, perturb_type)
    for road_folder in os.listdir(sub_log_path):  # roadGen_cutout_filter_road0_scale0_log

        print("Evaluation working on: ", road_folder, heatmap_type, summary_type,aggregation_method)
        match = re.search(r'_road(\d+)', road_folder)
        if match:
            road_number = match.group(1)
            print("Road Number is: ", )
        else:
            raise ValueError("The folder:", road_folder, "does not contain a road number.")

        evaluate_failure_prediction_roadGen(log_dir=log_dir,
                                            heatmap_type=heatmap_type,
                                            road_number=road_number,
                                            focus=focus,
                                            perturb_type=perturb_type,
                                            road_folder=road_folder,
                                            summary_type=summary_type,
                                            aggregation_method=aggregation_method
                                            )

def run_evaluation_tracks(aggregation_method, summary_type, heatmap_type, track, focus, save_merge):
    for perturb_folder in os.listdir(log_dir):
        if perturb_folder == f"{track}_normal":
            continue
        print("-------------------------------------------")
        print("Working on: ", perturb_folder, "Heatmap: ", heatmap_type, "Summary: ", summary_type, "Aggregation: ", aggregation_method)
        evaluate_failure_prediction(log_dir=log_dir,
                                    heatmap_type=heatmap_type,
                                    track=track,
                                    focus=focus,
                                    perturb_folder=perturb_folder,
                                    summary_type=summary_type,
                                    aggregation_method=aggregation_method,
                                    save_merge=save_merge,)


if __name__ == '__main__':

    heatmap_type_list = ["smooth_grad", "raw_smooth_grad",
                           "grad_cam_pp", "raw_grad_cam_pp",
                           "faster_score_cam", "raw_faster_score_cam",
                           "integrated_gradients", "raw_integrated_gradients",]
    # heatmap_type_list = ["integrated_gradients", "raw_integrated_gradients", ]

    aggregation_method_list = ['mean', 'max']

    track_list = ["lake", "mountain"]  # lake, mountain or roadGen
    focus_list = ["steer", "throttle"]  # steer, throttle

    thirdeye_summary_types = ['average_gradient', 'average']  # 'entropy', 'gini', 'topk', 'average_gradient', 'average']
    seye_summary_types = ['total_road_attention_ratio']

    summary_type_list = seye_summary_types

    # save printed logs
    log_path = "/home/jiaqq/Documents/ThirdEye-II/evaluate/general/threshold_crash_comparison_seye.log"
    log_file = open(log_path, 'w', encoding='utf-8')
    sys.stdout = log_file

    save_merge = False

    for aggregation_method in aggregation_method_list:
        for summary_type in summary_type_list:
            for heatmap_type in heatmap_type_list:
                for track in track_list:
                    for focus in focus_list:
                        log_dir = os.path.join("/data/ThirdEye-II/perturbationdrive/logs",
                                               "RoadGenerator" if track == "roadGen" else track)
                        if track == "roadGen":
                            run_evaluation_roadGen(aggregation_method, summary_type, heatmap_type, focus)
                        else:
                            run_evaluation_tracks(aggregation_method, summary_type, heatmap_type, track, focus, save_merge)

    sys.stdout = sys.__stdout__
    log_file.close()
import os
from evaluate_failure_prediction_heatmaps_scores import evaluate_failure_prediction

if __name__ == '__main__':

    heatmap_type_list = ["smooth_grad", "raw_smooth_grad",
                           "grad_cam_pp", "raw_grad_cam_pp",
                           "faster_score_cam", "raw_faster_score_cam",
                           "integrated_gradients", "raw_integrated_gradients", ]
    focus_list = ["steer", "throttle"]

    aggregation_method_list = ['mean', 'max']
    summary_types = ['average', 'average_gradient']

    track = "lake"
    log_dir = os.path.join("/home/jiaqq/Documents/ThirdEye-II/perturbationdrive/logs", track)
    for aggregation_method in aggregation_method_list:
        for summary_type in summary_types:
            for perturb_folder in os.listdir(log_dir):
                print(perturb_folder)
                evaluate_failure_prediction(log_dir=log_dir,
                                            heatmap_type="grad_cam_pp",
                                            track= track,
                                            focus="steer",
                                            perturb_folder=perturb_folder,
                                            summary_type=summary_type,
                                            aggregation_method=aggregation_method)
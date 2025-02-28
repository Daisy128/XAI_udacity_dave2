import os
# os.chdir("/home/jiaqq/Documents/ThirdEye-II/perturbationdrive/logs")
from evaluate_failure_prediction_heatmaps_scores import evaluate_failure_prediction

if __name__ == '__main__':
    track = "mountain"
    heatmap_method_list = ["smooth_grad", "raw_smooth_grad",
                           "grad_cam_pp", "raw_grad_cam_pp",
                           "faster_score_cam", "raw_faster_score_cam",
                           "integrated_gradients", "raw_integrated_gradients", ]

    ams=['mean', 'max']
    summary_types = ['average', 'average_gradient']
    log_dir = os.path.join("/home/jiaqq/Documents/ThirdEye-II/perturbationdrive/logs", track)
    # for perturb_type in os.listdir(log_dir):
    evaluate_failure_prediction(log_dir=log_dir,
                                heatmap_type="smooth_grad",
                                track= track,
                                focus="steer",
                                perturb_type='mountain_cutout_filter_scale5_log',
                                summary_type="average_gradient",
                                aggregation_method='max')
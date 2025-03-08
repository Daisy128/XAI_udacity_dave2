from evaluate_failure_prediction_heatmaps_scores import *

if __name__ == '__main__':

    heatmap_type_list = ["smooth_grad", "raw_smooth_grad",
                           "grad_cam_pp", "raw_grad_cam_pp",
                           "faster_score_cam", "raw_faster_score_cam",
                           "integrated_gradients", "raw_integrated_gradients",]
    # heatmap_type_list = ["integrated_gradients", "raw_integrated_gradients", ]
    focus_list = ["steer", "throttle"]

    aggregation_method_list = ['mean', 'max']
    # summary_types = ['average', 'average_gradient']
    summary_types = ['avg_road_attention_ratio'] #, ['total_road_attention_ratio','avg_road_attention_ratio']
    track = "mountain"
    focus = "throttle"
    log_dir = os.path.join("/home/jiaqq/Documents/ThirdEye-II/perturbationdrive/logs", track)
    # log_dir = os.path.join("/home/jiaqq/Documents/ThirdEye-II/perturbationdrive/logs", "test_seye")
    for aggregation_method in aggregation_method_list:
        for summary_type in summary_types:
            for heatmap_type in heatmap_type_list:
                for perturb_folder in os.listdir(log_dir):
                    if perturb_folder == f"{track}_normal":
                        continue
                    print("Evaluation working on: ", perturb_folder, heatmap_type, summary_type, aggregation_method)
                    evaluate_failure_prediction(log_dir=log_dir,
                                                heatmap_type=heatmap_type,
                                                track=track,
                                                focus=focus,
                                                perturb_folder=perturb_folder,
                                                summary_type=summary_type,
                                                aggregation_method=aggregation_method
                                                )

import csv
import gc
import glob
import os

import numpy as np
import pandas as pd
from keras import backend as K
from scipy.stats import gamma


def get_threshold(losses, conf_level=0.95):
    print("Fitting reconstruction error distribution using Gamma distribution")

    # removing zeros
    losses = np.array(losses)
    losses_copy = losses[losses != 0]
    shape, loc, scale = gamma.fit(losses_copy, floc=0)

    print("Creating threshold using the confidence intervals: %s" % conf_level)
    t = gamma.ppf(conf_level, shape, loc=loc, scale=scale)
    print('threshold: ' + str(t))
    return t

def load_heatmap_scores_segment(log_dir, heatmap_type, track, focus, perturb_folder, summary_type, aggregation_method):

    # 1. load heatmap scores in nominal conditions

    # Heatmap numpy scores when ADS drives normally, eg.:
    # /home/jiaqq/Documents/ThirdEye-II/perturbationdrive/logs/lake/lake_normal/smooth_grad_steer/smooth_grad_average_gradient_scores.npy
    path = os.path.join(log_dir,
                        f"{track}_normal",
                        f"{heatmap_type}_{focus}",
                        f"segment_{summary_type}.npy") # segment_total_road_attention_ratio.npy or segment_avg_road_attention_ratio.npy
    original_losses = np.load(path)

    new_original_losses = []
    for loss in original_losses:
        new_original_losses.append(1 - loss)
    original_losses = new_original_losses

    # different conditions in loading csv when segmentation

    path = glob.glob(f"{log_dir}/{track}_normal/{track}_normal.csv")
    data_df_nominal = pd.read_csv(path[0])

    data_df_nominal['loss'] = original_losses

    # 2. load heatmap scores in anomalous conditions

    path = os.path.join(log_dir,
                        perturb_folder,
                        f"{heatmap_type}_{focus}",
                        f"segment_{summary_type}.npy")
    anomalous_losses = np.load(path)

    path = glob.glob(f"{log_dir}/{perturb_folder}/*.csv")
    print(path[0])
    data_df_anomalous = pd.read_csv(path[0])

    new_anomalous_losses = []
    for loss in anomalous_losses:
        new_anomalous_losses.append(1 - loss)
    anomalous_losses = new_anomalous_losses

    data_df_anomalous['loss'] = anomalous_losses

    data_evaluate = {
        "heatmap_type": [heatmap_type],
        "summarization_method": [summary_type],
        "aggregation_method": [aggregation_method],
        "perturbation_folder": [perturb_folder],
        "focus": [focus]
    }

    return data_df_nominal, data_df_anomalous, data_evaluate

def load_heatmap_scores(log_dir, heatmap_type, track, focus, perturb_folder, summary_type, aggregation_method):

    # 1. load heatmap scores in nominal conditions

    path = os.path.join(log_dir,
                        f"{track}_normal",
                        f"{heatmap_type}_{focus}",
                        f"{heatmap_type}_{summary_type}_scores.npy")
    original_losses = np.load(path)

    path = glob.glob(f"{log_dir}/{track}_normal/{track}_normal.csv")
    data_df_nominal = pd.read_csv(path[0])

    data_df_nominal['loss'] = original_losses

    # 2. load heatmap scores in anomalous conditions

    path = os.path.join(log_dir,
                        perturb_folder,
                        f"{heatmap_type}_{focus}",
                        f"{heatmap_type}_{summary_type}_scores.npy")
    anomalous_losses = np.load(path)

    path = glob.glob(f"{log_dir}/{perturb_folder}/*.csv")
    print(path[0])
    data_df_anomalous = pd.read_csv(path[0])

    data_df_anomalous['loss'] = anomalous_losses

    data_evaluate = {
        "heatmap_type": [heatmap_type],
        "summarization_method": [summary_type],
        "aggregation_method": [aggregation_method],
        "perturbation_folder": [perturb_folder],
        "focus": [focus]
    }

    return data_df_nominal, data_df_anomalous, data_evaluate

def evaluate_failure_prediction(log_dir, heatmap_type, track, focus, perturb_folder, summary_type, aggregation_method):

    # 1. load data

    data_df_nominal_hm = data_df_anomalous_hm = data_evaluate_hm = None
    data_df_nominal_seg = data_df_anomalous_seg = data_evaluate_seg = None


    data_df_nominal_hm, data_df_anomalous_hm, data_evaluate_hm = load_heatmap_scores(log_dir=log_dir,
                                                                            heatmap_type="smooth_grad",
                                                                            track=track,
                                                                            focus="steer",
                                                                            perturb_folder=perturb_folder,
                                                                            summary_type="total_road_attention_ratio",
                                                                            aggregation_method='max')

    if summary_type in {'total_road_attention_ratio', 'avg_road_attention_ratio'}:
        data_df_nominal_seg, data_df_anomalous_seg, data_evaluate_seg = load_heatmap_scores_segment(log_dir=log_dir,
                                                                                                    heatmap_type="smooth_grad",
                                                                                                    track=track,
                                                                                                    focus="steer",
                                                                                                    perturb_folder=perturb_folder,
                                                                                                    summary_type="total_road_attention_ratio",
                                                                                                    aggregation_method='max')

    # 2. compute a threshold from nominal conditions, and FP and TN

    threshold_hm = threshold_seg = None

    false_positive_windows, true_negative_windows, threshold_hm = compute_fp_and_tn(data_df_nominal_hm,
                                                                                    data_evaluate_hm['aggregation_method'][0])
    if summary_type in {'total_road_attention_ratio', 'avg_road_attention_ratio'}:
        _, _, threshold_seg = compute_fp_and_tn(data_df_nominal_seg,
                                            data_evaluate_seg['aggregation_method'][0])

    # 3. compute TP and FN using different time to misbehaviour windows in different value of reaction windows
    for seconds in range(1, 4):
        true_positive_windows, false_negative_windows, undetectable_windows = compute_tp_and_fn(data_df_anomalous_hm,
                                                                                                data_df_anomalous_hm['loss'],
                                                                                                threshold_hm,
                                                                                                seconds,
                                                                                                data_evaluate_hm['aggregation_method'][0],
                                                                                                data_df_anomalous_seg,
                                                                                                data_df_anomalous_seg['loss'],
                                                                                                threshold_seg)

        if true_positive_windows != 0:
            precision = true_positive_windows / (true_positive_windows + false_positive_windows)
            recall = true_positive_windows / (true_positive_windows + false_negative_windows)
            accuracy = (true_positive_windows + true_negative_windows) / (
                    true_positive_windows + true_negative_windows + false_positive_windows + false_negative_windows)
            fpr = false_positive_windows / (false_positive_windows + true_negative_windows)

            if precision != 0 or recall != 0:
                f3 = true_positive_windows / (
                        true_positive_windows + 0.1 * false_positive_windows + 0.9 * false_negative_windows)
            else:
                precision = recall = f3 = accuracy = fpr = 0
        else:
            precision = recall = f3 = accuracy = fpr = 0

        # 4. write results in a CSV files
        if data_df_anomalous_seg is not None:
            evaluation_data = data_evaluate_seg.copy()
            csv_filename = f"{evaluation_data['heatmap_type'][0]}_{evaluation_data['focus'][0]}_seye.csv"
        else:
            evaluation_data = data_evaluate_hm.copy()
            csv_filename = f"{evaluation_data['heatmap_type'][0]}_{evaluation_data['focus'][0]}_thirdeye.csv"

        evaluation_data.update({
            "failures": [true_positive_windows + false_negative_windows],
            "detected": [true_positive_windows],
            "undetected": [false_negative_windows],
            "undetectable": [undetectable_windows],
            "ttm": [seconds],
            "accuracy": [round(accuracy * 100)],
            "fpr": [round(fpr * 100)],
            "precision": [round(precision * 100)],
            "recall": [round(recall * 100)],
            "f3": [round(f3 * 100)]
        })

        df = pd.DataFrame(evaluation_data)

        if not os.path.exists(csv_filename):
            df.to_csv(csv_filename, index=False)
        else:
            # 'a' extends csv
            df.to_csv(csv_filename, mode='a', index=False, header=False)

    K.clear_session()
    gc.collect()


def compute_tp_and_fn(data_df_anomalous_hm, losses_on_anomalous_hm, threshold_hm, seconds_to_anticipate,
                      aggregation_method='mean', data_df_anomalous_seg=None, losses_on_anomalous_seg=None, threshold_seg=None):
    print("time to misbehaviour (s): %d" % seconds_to_anticipate)

    # only occurring when conditions == unexpected
    true_positive_windows = 0
    false_negative_windows = 0
    undetectable_windows = 0

    # 总帧数： data index最大值，帧率10
    number_frames_anomalous = pd.Series.max(data_df_anomalous_hm['index'])
    fps_anomalous = 10

    crashed_anomalous = data_df_anomalous_hm['is_crashed']
    crashed_anomalous.is_copy = None
    crashed_anomalous_in_anomalous_conditions = crashed_anomalous.copy()

    # creates the ground truth，找出序列中第一个出现True的帧位置
    all_first_frame_position_crashed_sequences = []
    for idx, item in enumerate(crashed_anomalous_in_anomalous_conditions):
        if idx == number_frames_anomalous:  # we have reached the end of the file
            continue
        # Since I used False/True to represent crashed or not, instead of 0/1
        if not item and crashed_anomalous_in_anomalous_conditions[idx + 1]:
            first_index_crash = idx + 1
            all_first_frame_position_crashed_sequences.append(first_index_crash)

    # 识别到多少次事故以及各自的首帧位置
    print("identified %d crash(es)" % len(all_first_frame_position_crashed_sequences))
    print(all_first_frame_position_crashed_sequences)

    # 如提前3秒识别，意味着提前45帧识别
    frames_to_reassign = fps_anomalous * seconds_to_anticipate  # start of the sequence

    # frames_to_reassign_2 = 1  # first frame before the failure
    # 识别窗口应为事故帧往前数45帧到30帧之间
    frames_to_reassign_2 = fps_anomalous * (seconds_to_anticipate - 1)  # first frame n seconds before the failure

    reaction_window = pd.Series()

    # 遍历每个事故
    for item in all_first_frame_position_crashed_sequences:
        print("analysing failure %d" % item)
        # 说明事故发生太早，没有足够帧数构成预警窗口
        if item - frames_to_reassign < 0:
            undetectable_windows += 1
            continue

        # the detection window overlaps with a previous crash; skip it
        # 窗口与前一次事故重叠
        if crashed_anomalous_in_anomalous_conditions.loc[
           item - frames_to_reassign: item - frames_to_reassign_2].sum() > 2:
            print("failure %d cannot be detected at TTM=%d" % (item, seconds_to_anticipate))
            undetectable_windows += 1

        # 将crashed_anomalous_in_anomalous_conditions设为True表示这是监测窗口
        else:
            crashed_anomalous_in_anomalous_conditions.loc[item - frames_to_reassign: item - frames_to_reassign_2] = True

            reaction_window = pd.concat([reaction_window, crashed_anomalous_in_anomalous_conditions[item - frames_to_reassign: item - frames_to_reassign_2]])
            print("frames between %d and %d have been labelled as 1" % (
                item - frames_to_reassign, item - frames_to_reassign_2))
            print("reaction frames size is %d" % len(reaction_window))

            # sma_anomalous来自heatmap_scores，并平滑均值？
            sma_anomalous_hm = pd.Series(losses_on_anomalous_hm)
            sma_anomalous_hm = sma_anomalous_hm.iloc[reaction_window.index.to_list()]
            assert len(reaction_window) == len(sma_anomalous_hm)

            if losses_on_anomalous_seg is not None:
                sma_anomalous_seg = pd.Series(losses_on_anomalous_seg)
                sma_anomalous_seg = sma_anomalous_seg.iloc[reaction_window.index.to_list()]
                assert len(reaction_window) == len(sma_anomalous_seg)
            else:
                sma_anomalous_seg = None

            # 计算窗口分数
            def aggregate(series, method):
                return series.mean() if method == 'mean' else series.max()

            aggregated_score_hm = aggregate(sma_anomalous_hm, aggregation_method)
            if sma_anomalous_seg is not None:
                aggregated_score_seg = aggregate(sma_anomalous_seg, aggregation_method)
            else:
                aggregated_score_seg = None

            # aggregated_score = None
            # if aggregation_method == "mean":
            #     aggregated_score = sma_anomalous.mean()
            # elif aggregation_method == "max":
            #     aggregated_score = sma_anomalous.max()

            # print("threshold %s\tmean: %s\tmax: %s" % (
            #     str(threshold), str(sma_anomalous.mean()), str(sma_anomalous.max())))

            if aggregated_score_hm >= threshold_hm or (threshold_seg is not None and aggregated_score_seg >= threshold_seg):
                true_positive_windows += 1
            elif aggregated_score_hm < threshold_hm or (threshold_seg is not None and aggregated_score_seg < threshold_seg):
                false_negative_windows += 1

        print("failure: %d - true positives: %d - false negatives: %d - undetectable: %d" % (
            item, true_positive_windows, false_negative_windows, undetectable_windows))

    assert len(all_first_frame_position_crashed_sequences) == (
            true_positive_windows + false_negative_windows + undetectable_windows)
    return true_positive_windows, false_negative_windows, undetectable_windows


def compute_fp_and_tn(data_df_nominal, aggregation_method):
    # when conditions == nominal I count only FP and TN

    # 帧率fps_nominal = 数据集的总帧数 / 总时间 = 窗口长度
    # number_frames_nominal = pd.Series.max(data_df_nominal['index'])
    # simulation_time_nominal = pd.Series.max(data_df_nominal['time']) # ？ time not logged
    # fps_nominal = number_frames_nominal // simulation_time_nominal
    fps_nominal = 10  # TODO： 对于nominal 帧率在『10,9,11』循环

    # 窗口数量 = 数据集总长度 / 窗口长度
    num_windows_nominal = len(data_df_nominal) // fps_nominal
    # 删除余数
    if len(data_df_nominal) % fps_nominal != 0:
        num_to_delete = len(data_df_nominal) - (num_windows_nominal * fps_nominal) - 1
        data_df_nominal = data_df_nominal[:-num_to_delete]

    # data_df_nominal['loss']即 np.load(heatmap-scores.npy)
    losses = pd.Series(data_df_nominal['loss'])
    # losses.rolling.mean() 滑动并计算平均， 长度为窗口长度
    sma_nominal = losses.rolling(fps_nominal, min_periods=1).mean()

    list_aggregated = []

    for idx, loss in enumerate(sma_nominal):

        if idx > 0 and idx % fps_nominal == 0:
            aggregated_score = None
            if aggregation_method == "mean":
                # 时间窗口内的滑动平均值的平均分数
                # .iloc获取从 (idx - fps_nominal) 到 idx 之间（10个值）的数据
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).mean()

            elif aggregation_method == "max":
                # 时间窗口内的滑动平均值的最大分数
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).max()

            list_aggregated.append(aggregated_score)

        elif idx == len(sma_nominal) - 1:

            aggregated_score = None
            if aggregation_method == "mean":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).mean()
            elif aggregation_method == "max":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).max()

            list_aggregated.append(aggregated_score)

    assert len(list_aggregated) == num_windows_nominal
    # confidence level 95%
    threshold = get_threshold(list_aggregated, conf_level=0.95)

    false_positive_windows = len([i for i in list_aggregated if i > threshold])
    true_negative_windows = len([i for i in list_aggregated if i <= threshold])

    assert false_positive_windows + true_negative_windows == num_windows_nominal
    print("false positives: %d - true negatives: %d" % (false_positive_windows, true_negative_windows))

    return false_positive_windows, true_negative_windows, threshold

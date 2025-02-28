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


def evaluate_failure_prediction(log_dir, heatmap_type, track, focus, perturb_type, summary_type, aggregation_method):
    # print("Using summarization average" if summary_type is '-avg' else "Using summarization gradient")
    # print("Using aggregation mean" if aggregation_method is 'mean' else "Using aggregation max")

    # 1. load heatmap scores in nominal conditions

    path = os.path.join(log_dir,
                        f"{track}_normal",
                        f"{heatmap_type}_{focus}",
                        f"{heatmap_type}_{summary_type}_scores.npy")
    original_losses = np.load(path)

    path = glob.glob(f"{log_dir}/{track}_normal/*.csv")
    data_df_nominal = pd.read_csv(path[0])

    data_df_nominal['loss'] = original_losses

    # 2. load heatmap scores in anomalous conditions

    path = os.path.join(log_dir,
                        perturb_type,
                        f"{heatmap_type}_{focus}",
                        f"{heatmap_type}_{summary_type}_scores.npy")
    anomalous_losses = np.load(path)

    path = glob.glob(f"{log_dir}/{perturb_type}/*.csv")
    print(path[0])
    data_df_anomalous = pd.read_csv(path[0])

    data_df_anomalous['loss'] = anomalous_losses

    # 3. compute a threshold from nominal conditions, and FP and TN
    false_positive_windows, true_negative_windows, threshold = compute_fp_and_tn(data_df_nominal, aggregation_method)

    # 4. compute TP and FN using different time to misbehaviour windows
    for seconds in range(1, 4):
        true_positive_windows, false_negative_windows, undetectable_windows = compute_tp_and_fn(data_df_anomalous,
                                                                                                anomalous_losses,
                                                                                                threshold,
                                                                                                seconds,
                                                                                                aggregation_method)

        if true_positive_windows != 0:
            precision = true_positive_windows / (true_positive_windows + false_positive_windows)
            recall = true_positive_windows / (true_positive_windows + false_negative_windows)
            accuracy = (true_positive_windows + true_negative_windows) / (
                    true_positive_windows + true_negative_windows + false_positive_windows + false_negative_windows)
            fpr = false_positive_windows / (false_positive_windows + true_negative_windows)

            if precision != 0 or recall != 0:
                f3 = true_positive_windows / (
                        true_positive_windows + 0.1 * false_positive_windows + 0.9 * false_negative_windows)

                print("Accuracy: " + str(round(accuracy * 100)) + "%")
                print("False Positive Rate: " + str(round(fpr * 100)) + "%")
                print("Precision: " + str(round(precision * 100)) + "%")
                print("Recall: " + str(round(recall * 100)) + "%")
                print("F-3: " + str(round(f3 * 100)) + "%\n")
            else:
                precision = recall = f3 = accuracy = fpr = 0
                print("Accuracy: undefined")
                print("False Positive Rate: undefined")
                print("Precision: undefined")
                print("Recall: undefined")
                print("F-3: undefined\n")
        else:
            precision = recall = f3 = accuracy = fpr = 0
            print("Accuracy: undefined")
            print("False Positive Rate: undefined")
            print("Precision: undefined")
            print("Recall: undefined")
            print("F-1: undefined")
            print("F-3: undefined\n")

        # 5. write results in a CSV files
        if not os.path.exists(heatmap_type + '_' + focus + '.csv'):
            with open(heatmap_type + '_' + focus + '.csv', mode='w',
                      newline='') as result_file:
                writer = csv.writer(result_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                writer.writerow(
                    ["heatmap_type", "summarization_method", "aggregation_type", "simulation_name", "failures",
                     "detected", "undetected", "undetectable", "ttm", 'accuracy', "fpr", "precision", "recall",
                     "f3"])
                writer.writerow([heatmap_type, summary_type[1:], aggregation_method, perturb_type,
                                 str(true_positive_windows + false_negative_windows),
                                 str(true_positive_windows),
                                 str(false_negative_windows),
                                 str(undetectable_windows),
                                 seconds,
                                 str(round(accuracy * 100)),
                                 str(round(fpr * 100)),
                                 str(round(precision * 100)),
                                 str(round(recall * 100)),
                                 str(round(f3 * 100))])

        else:
            with open(heatmap_type + '-' + focus + '.csv', mode='a',
                      newline='') as result_file:
                writer = csv.writer(result_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                writer.writerow([heatmap_type, summary_type[1:], aggregation_method, perturb_type,
                                 str(true_positive_windows + false_negative_windows),
                                 str(true_positive_windows),
                                 str(false_negative_windows),
                                 str(undetectable_windows),
                                 seconds,
                                 str(round(accuracy * 100)),
                                 str(round(fpr * 100)),
                                 str(round(precision * 100)),
                                 str(round(recall * 100)),
                                 str(round(f3 * 100))])

    K.clear_session()
    gc.collect()


def compute_tp_and_fn(data_df_anomalous, losses_on_anomalous, threshold, seconds_to_anticipate,
                      aggregation_method='mean'):
    print("time to misbehaviour (s): %d" % seconds_to_anticipate)

    # only occurring when conditions == unexpected
    true_positive_windows = 0
    false_negative_windows = 0
    undetectable_windows = 0

    ''' 
        prepare dataset to get TP and FN from unexpected
        '''
    # number_frames_anomalous = pd.Series.max(data_df_anomalous['frameId']) # im
    # simulation_time_anomalous = pd.Series.max(data_df_anomalous['time']) # im
    # fps_anomalous = number_frames_anomalous // simulation_time_anomalous
    number_frames_anomalous = pd.Series.max(data_df_anomalous['index'])  # im
    fps_anomalous = 15

    crashed_anomalous = data_df_anomalous['is_crashed'] # im
    crashed_anomalous.is_copy = None
    crashed_anomalous_in_anomalous_conditions = crashed_anomalous.copy()

    # creates the ground truth
    all_first_frame_position_crashed_sequences = []
    for idx, item in enumerate(crashed_anomalous_in_anomalous_conditions):
        if idx == number_frames_anomalous - 1:  # we have reached the end of the file
            continue

        # if crashed_anomalous_in_anomalous_conditions[idx] == 0 and crashed_anomalous_in_anomalous_conditions[
        #     idx + 1] == 1:

        # Since I used False/True to represent crashed or not, instead of 0/1
        if not item and crashed_anomalous_in_anomalous_conditions[idx + 1]:
            first_index_crash = idx + 1
            all_first_frame_position_crashed_sequences.append(first_index_crash)
            # print("first_index_crash: %d" % first_index_crash)

    print("identified %d crash(es)" % len(all_first_frame_position_crashed_sequences))
    print(all_first_frame_position_crashed_sequences)
    frames_to_reassign = fps_anomalous * seconds_to_anticipate  # start of the sequence

    # frames_to_reassign_2 = 1  # first frame before the failure
    frames_to_reassign_2 = fps_anomalous * (seconds_to_anticipate - 1)  # first frame n seconds before the failure

    reaction_window = pd.Series()

    for item in all_first_frame_position_crashed_sequences:
        print("analysing failure %d" % item)
        if item - frames_to_reassign < 0:
            undetectable_windows += 1
            continue

        # the detection window overlaps with a previous crash; skip it
        if crashed_anomalous_in_anomalous_conditions.loc[
           item - frames_to_reassign: item - frames_to_reassign_2].sum() > 2:
            print("failure %d cannot be detected at TTM=%d" % (item, seconds_to_anticipate))
            undetectable_windows += 1
        else:
            crashed_anomalous_in_anomalous_conditions.loc[item - frames_to_reassign: item - frames_to_reassign_2] = True

            reaction_window = pd.concat([reaction_window, crashed_anomalous_in_anomalous_conditions[item - frames_to_reassign: item - frames_to_reassign_2]])
            print("frames between %d and %d have been labelled as 1" % (
                item - frames_to_reassign, item - frames_to_reassign_2))
            print("reaction frames size is %d" % len(reaction_window))

            sma_anomalous = pd.Series(losses_on_anomalous)
            sma_anomalous = sma_anomalous.iloc[reaction_window.index.to_list()]
            assert len(reaction_window) == len(sma_anomalous)

            # print(sma_anomalous)

            aggregated_score = None
            if aggregation_method == "mean":
                aggregated_score = sma_anomalous.mean()
            elif aggregation_method == "max":
                aggregated_score = sma_anomalous.max()

            print("threshold %s\tmean: %s\tmax: %s" % (
                str(threshold), str(sma_anomalous.mean()), str(sma_anomalous.max())))

            if aggregated_score >= threshold:
                true_positive_windows += 1
            elif aggregated_score < threshold:
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
    # simulation_time_nominal = pd.Series.max(data_df_nominal['time'])
    # fps_nominal = number_frames_nominal // simulation_time_nominal
    fps_nominal = 15

    # 窗口数量 = 数据集总长度 / 窗口长度， 删除余数
    num_windows_nominal = len(data_df_nominal) // fps_nominal
    if len(data_df_nominal) % fps_nominal != 0:
        num_to_delete = len(data_df_nominal) - (num_windows_nominal * fps_nominal) - 1
        data_df_nominal = data_df_nominal[:-num_to_delete]

    # data_df_nominal['loss']即 np.load(heatmap-scores.npy)
    losses = pd.Series(data_df_nominal['loss'])
    # losses.rolling.mean() 滑动平均， 长度为窗口长度
    sma_nominal = losses.rolling(fps_nominal, min_periods=1).mean()

    list_aggregated = []

    for idx, loss in enumerate(sma_nominal):

        if idx > 0 and idx % fps_nominal == 0:

            aggregated_score = None
            if aggregation_method == "mean":
                # 时间窗口内的滑动平均值的平均分数
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

    # 得分超过阈值的窗口
    false_positive_windows = len([i for i in list_aggregated if i > threshold])
    # 得分不超过阈值的窗口
    true_negative_windows = len([i for i in list_aggregated if i <= threshold])

    assert false_positive_windows + true_negative_windows == num_windows_nominal
    print("false positives: %d - true negatives: %d" % (false_positive_windows, true_negative_windows))

    return false_positive_windows, true_negative_windows, threshold

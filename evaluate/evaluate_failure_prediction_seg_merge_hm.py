import csv
import gc
import os

import numpy as np
import pandas as pd
from keras import backend as K
from scipy.stats import gamma


def get_threshold(losses, conf_level):
    print("Fitting reconstruction error distribution using Gamma distribution")

    # removing zeros
    losses = np.array(losses)
    losses_copy = losses[losses != 0]
    shape, loc, scale = gamma.fit(losses_copy, floc=0)

    print("Creating threshold using the confidence intervals: %s" % conf_level)
    t = gamma.ppf(conf_level, shape, loc=loc, scale=scale)
    print('threshold: ' + str(t))
    return t


def evaluate_failure_prediction(cfg, heatmap_type, simulation_name, summary_type, aggregation_method, condition,
                                segmentation):
    print("Using summarization average" if summary_type is '-avg' else "Using summarization gradient")
    print("Using aggregation mean" if aggregation_method is 'mean' else "Using aggregation max")

    # 1. load heatmap scores in nominal conditions

    cfg.SIMULATION_NAME = 'gauss-journal-track1-nominal'
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'htm-' + heatmap_type + '-scores' + summary_type + '.npy')

    original_losses = np.load(path)

    if summary_type == '-total_road_attention_percentage' or summary_type == 'road_percentage':  # Why?
        new_original_losses = []
        for loss in original_losses:
            new_original_losses.append(1 - loss)
        original_losses = new_original_losses

    if segmentation: # why different csv logs?
        if heatmap_type == 'smoothgrad':
            path = os.path.join(cfg.TESTING_DATA_DIR,
                                cfg.SIMULATION_NAME,
                                'segmentation-SmoothGrad',
                                'driving_log.csv')
        elif heatmap_type == 'faster-scorecam':
            path = os.path.join(cfg.TESTING_DATA_DIR,
                                cfg.SIMULATION_NAME,
                                'segmentation-Faster-ScoreCAM',
                                'driving_log.csv')
    else:
        path = os.path.join(cfg.TESTING_DATA_DIR,
                            cfg.SIMULATION_NAME,
                            'heatmaps-' + heatmap_type,
                            'driving_log.csv')

    data_df_nominal = pd.read_csv(path)

    data_df_nominal['loss'] = original_losses

    # 1.2 seperate load hm # why specifc loading smoothgrad & avg-grad again?
    cfg.SIMULATION_NAME = 'gauss-journal-track1-nominal'
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'htm-' + 'smoothgrad' + '-scores' + '-avg-grad' + '.npy')

    original_losses_hm = np.load(path)

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'heatmaps-' + 'smoothgrad',
                        'driving_log.csv')

    data_df_nominal_hm = pd.read_csv(path)

    data_df_nominal_hm['loss'] = original_losses_hm

    # 2. load heatmap scores in anomalous conditions

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        simulation_name,
                        'htm-' + heatmap_type + '-scores' + summary_type + '.npy')
    anomalous_losses = np.load(path)

    new_anomalous_losses = []

    if summary_type == '-total_road_attention_percentage' or summary_type == 'road_percentage':
        for loss in anomalous_losses:
            new_anomalous_losses.append(1 - loss)

        anomalous_losses = new_anomalous_losses

    # TODO do for segmentation-Smoothgrad and segmentation-Faster-ScoreCAM

    if segmentation:
        if heatmap_type == 'smoothgrad':
            path = os.path.join(cfg.TESTING_DATA_DIR,
                                simulation_name,
                                'segmentation-SmoothGrad',
                                'driving_log.csv')
        elif heatmap_type == 'faster-scorecam':
            path = os.path.join(cfg.TESTING_DATA_DIR,
                                simulation_name,
                                'segmentation-Faster-ScoreCAM',
                                'driving_log.csv')
    else:
        path = os.path.join(cfg.TESTING_DATA_DIR,
                            simulation_name,
                            'heatmaps-' + heatmap_type,
                            'driving_log.csv')

    data_df_anomalous = pd.read_csv(path)

    # data_df_anomalous = temp[temp['avg_all_attention'] != 0]

    data_df_anomalous['loss'] = anomalous_losses

    # 2.2load hm version
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        simulation_name,
                        'htm-' + 'smoothgrad' + '-scores' + '-avg-grad' + '.npy')
    anomalous_losses_hm = np.load(path)

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        simulation_name,
                        'heatmaps-' + 'smoothgrad',
                        'driving_log.csv')

    data_df_anomalous_hm = pd.read_csv(path)

    # data_df_anomalous = temp[temp['avg_all_attention'] != 0]

    data_df_anomalous_hm['loss'] = anomalous_losses_hm

    # 3. compute a threshold from nominal conditions, and FP and TN
    false_positive_windows, true_negative_windows, threshold = compute_fp_and_tn(data_df_nominal,
                                                                                 aggregation_method,
                                                                                 condition)

    _, _, threshold_hm = compute_fp_and_tn(data_df_nominal_hm,
                                           aggregation_method,
                                           condition)

    # 4. compute TP and FN using different time to misbehaviour windows
    for seconds in range(1, 4):
        true_positive_windows, false_negative_windows, undetectable_windows = compute_tp_and_fn(data_df_anomalous,
                                                                                                anomalous_losses,
                                                                                                anomalous_losses_hm,
                                                                                                threshold,
                                                                                                threshold_hm,
                                                                                                seconds,
                                                                                                aggregation_method,)

        if true_positive_windows != 0:
            precision = true_positive_windows / (
                    true_positive_windows + false_positive_windows)  # when the model predicts a window as positive, it is very likely to be correct. when high
            recall = true_positive_windows / (
                    true_positive_windows + false_negative_windows)  # Recall relates to your ability to detect the positive cases -low recall missing cases
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
        if not os.path.exists(heatmap_type + '-' + str(condition) + '.csv'):
            with open(heatmap_type + '-' + str(condition) + '.csv', mode='w',
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
                writer.writerow([heatmap_type, 'seg_merge_hm', aggregation_method, simulation_name,
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
            with open(heatmap_type + '-' + str(condition) + '.csv', mode='a',
                      newline='') as result_file:
                writer = csv.writer(result_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                writer.writerow([heatmap_type, 'seg_merge_hm', aggregation_method, simulation_name,
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


def compute_tp_and_fn(data_df_anomalous, losses_on_anomalous, losses_on_anomalous_hm, threshold,
                      threshold_hm, seconds_to_anticipate,
                      aggregation_method='mean', cond='ood'):
    print("time to misbehaviour (s): %d" % seconds_to_anticipate)

    # only occurring when conditions == unexpected
    true_positive_windows = 0
    false_negative_windows = 0
    undetectable_windows = 0

    ''' 
        prepare dataset to get TP and FN from unexpected
        '''
    if cond == "icse20":
        number_frames_anomalous = pd.Series.max(data_df_anomalous['frameId'])
        fps_anomalous = 15  # only for icse20 configurations

    else:
        number_frames_anomalous = pd.Series.max(data_df_anomalous['frameId'])
        simulation_time_anomalous = pd.Series.max(data_df_anomalous['time'])
        fps_anomalous = number_frames_anomalous // simulation_time_anomalous

    crashed_anomalous = data_df_anomalous['crashed']
    crashed_anomalous.is_copy = None
    crashed_anomalous_in_anomalous_conditions = crashed_anomalous.copy()

    # creates the ground truth
    all_first_frame_position_crashed_sequences = []
    for idx, item in enumerate(crashed_anomalous_in_anomalous_conditions):
        if idx == number_frames_anomalous:  # we have reached the end of the file
            continue

        if crashed_anomalous_in_anomalous_conditions[idx] == 0 and crashed_anomalous_in_anomalous_conditions[
            idx + 1] == 1:
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
            crashed_anomalous_in_anomalous_conditions.loc[item - frames_to_reassign: item - frames_to_reassign_2] = 1
            reaction_window = reaction_window.append(
                crashed_anomalous_in_anomalous_conditions[item - frames_to_reassign: item - frames_to_reassign_2])

            print("frames between %d and %d have been labelled as 1" % (
                item - frames_to_reassign, item - frames_to_reassign_2))
            print("reaction frames size is %d" % len(reaction_window))

            sma_anomalous = pd.Series(losses_on_anomalous)
            sma_anomalous = sma_anomalous.iloc[reaction_window.index.to_list()]
            assert len(reaction_window) == len(sma_anomalous)

            # seperate for hm
            sma_anomalous_hm = pd.Series(losses_on_anomalous_hm)
            sma_anomalous_hm = sma_anomalous_hm.iloc[reaction_window.index.to_list()]
            assert len(reaction_window) == len(sma_anomalous_hm)

            # print(sma_anomalous)

            aggregated_score = None
            aggregated_score_hm = None
            if aggregation_method == "mean":
                aggregated_score = sma_anomalous.mean()
                aggregated_score_hm = sma_anomalous_hm.mean()

            elif aggregation_method == "max":
                aggregated_score = sma_anomalous.max()
                aggregated_score_hm = sma_anomalous_hm.max()

            print("threshold %s\tmean: %s\tmax: %s" % (
                str(threshold), str(sma_anomalous.mean()), str(sma_anomalous.max())))

            # Core reason why hm is added in evaluation
            if aggregated_score >= threshold or aggregated_score_hm >= threshold_hm:
                true_positive_windows += 1
            elif aggregated_score < threshold or aggregated_score_hm < threshold_hm:
                false_negative_windows += 1

        print("failure: %d - true positives: %d - false negatives: %d - undetectable: %d" % (
            item, true_positive_windows, false_negative_windows, undetectable_windows))

    assert len(all_first_frame_position_crashed_sequences) == (
            true_positive_windows + false_negative_windows + undetectable_windows)
    return true_positive_windows, false_negative_windows, undetectable_windows


def compute_fp_and_tn(data_df_nominal, aggregation_method, condition):
    # when conditions == nominal I count only FP and TN

    if condition == "icse20":
        fps_nominal = 15  # only for icse20 configurations
    else:
        number_frames_nominal = pd.Series.max(data_df_nominal['frameId'])
        simulation_time_nominal = pd.Series.max(data_df_nominal['time'])
        fps_nominal = number_frames_nominal // simulation_time_nominal

    num_windows_nominal = len(data_df_nominal) // fps_nominal
    if len(data_df_nominal) % fps_nominal != 0:
        num_to_delete = len(data_df_nominal) - (num_windows_nominal * fps_nominal) - 1
        data_df_nominal = data_df_nominal[:-num_to_delete]

    losses = pd.Series(data_df_nominal['loss'])
    sma_nominal = losses.rolling(fps_nominal, min_periods=1).mean()

    list_aggregated = []

    for idx, loss in enumerate(sma_nominal):

        if idx > 0 and idx % fps_nominal == 0:

            aggregated_score = None
            if aggregation_method == "mean":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).mean()

            elif aggregation_method == "max":
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
    threshold = get_threshold(list_aggregated, conf_level=0.95)

    false_positive_windows = len([i for i in list_aggregated if i > threshold])
    true_negative_windows = len([i for i in list_aggregated if i <= threshold])

    assert false_positive_windows + true_negative_windows == num_windows_nominal
    print("false positives: %d - true negatives: %d" % (false_positive_windows, true_negative_windows))

    return false_positive_windows, true_negative_windows, threshold


if __name__ == '__main__':
    #compute metrix
    for path in ['C:/Unet/ThirdEye/ase22/xai/faster-scorecam-mutants.csv'
                 ]:  # 'C:/Unet/ThirdEye/ase22/xai/smoothgrad-icse20.csv',
        # path = 'C:/Unet/ThirdEye/ase22/xai/faster-scorecam-mutants.csv'
        data_df = pd.read_csv(path)

        for agg_type in ['max', 'mean']:
            for sum_method in ['road_percentage', 'avg_road_attention_percentage',
                               'total_road_attention_percentage']:  # , 'avg-grad',
                filtered_df = data_df[(data_df['summarization_method'] == sum_method) &
                                      (data_df['aggregation_type'] == agg_type)]
                # & (data_df['ttm'] == 3)]

                average_f3 = filtered_df['f3'].mean()
                Pr = filtered_df['precision'].mean()
                Re = filtered_df['recall'].mean()

                print(
                    f" '{path}' with summarization_method is '{sum_method}' and aggregation_type is '{agg_type}' '{Pr}', '{Re}', '{average_f3}'"
                )

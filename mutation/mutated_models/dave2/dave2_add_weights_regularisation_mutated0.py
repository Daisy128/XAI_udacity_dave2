
import os
from mutation.operators import weights_operators
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import timedelta, datetime
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from utils.conf import track_infos, CHECKPOINT_DIR, Training_Configs, model_cfgs, mutate_cfgs
from model.lane_keeping.self_driving_car_batch_generator import Generator
from model.lane_keeping.lane_keeping_models_tf import build_model
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def load_data(track_index):
    '\n    Load training data_nominal and split it into training and validation set\n    '
    track_info = track_infos[track_index]
    driving_styles = track_info['driving_style']
    print((('Loading training set ' + str(track_info['track_name'])) + str(driving_styles)))
    start = time.time()
    x_list = []
    y_steering_list = []
    y_throttle_list = []
    column_name = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed', 'lap', 'sector', 'cte']
    for drive_style in driving_styles:
        try:
            csv_path = os.path.join(track_info['training_data_dir'], drive_style, 'driving_log.csv')
            data_df = pd.read_csv(csv_path, header=0)
            if (list(data_df.columns) != column_name):
                data_df.columns = column_name
            if Training_Configs['AUG']['USE_LEFT_RIGHT']:
                y_throttle_center = data_df['throttle'].values
                y_throttle_left = (y_throttle_center / 1.2)
                y_throttle_right = (y_throttle_center / 1.2)
                y_center = data_df['steering'].values
                y_left = (y_center + 0.1)
                y_right = (y_center - 0.1)
                new_x = np.concatenate([data_df['center'].values, data_df['left'].values, data_df['right'].values])
                new_y_steering = np.concatenate([y_center, y_left, y_right])
                new_y_throttle = np.concatenate([y_throttle_center, y_throttle_left, y_throttle_right])
                x_list.append(new_x)
                y_steering_list.append(new_y_steering)
                y_throttle_list.append(new_y_throttle)
            else:
                y_throttle_center = data_df['throttle'].values
                y_steering_center = data_df['steering'].values
                x_center = data_df['center'].values
                x_list.append(x_center)
                y_steering_list.append(y_steering_center)
                y_throttle_list.append(y_throttle_center)
        except FileNotFoundError:
            print(('Unable to read file %s' % csv_path))
            continue
    if (not x_list):
        print('No driving data were provided for training. Provide correct paths to the driving_log.csv files.')
        exit()
    x = np.concatenate(x_list, axis=0)
    y_steering = np.concatenate(y_steering_list, axis=0).reshape((- 1), 1)
    y_throttle = np.concatenate(y_throttle_list, axis=0).reshape((- 1), 1)
    y = np.concatenate((y_steering, y_throttle), axis=1)
    try:
        (x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=Training_Configs['TEST_SIZE'], random_state=0)
    except TypeError:
        print('Missing header to csv files')
        exit()
    duration_train = (time.time() - start)
    print(('Loading training set completed in %s.' % str(timedelta(seconds=round(duration_train)))))
    print(f'Data set: {len(x)} elements')
    print(f'Training set: {len(x_train)} elements')
    print(f'Test set: {len(x_test)} elements')
    return (x_train, x_test, y_train, y_test)

def get_generators(x_train, x_test, y_train, y_test):
    if Training_Configs['SHUFFLE_DATA']:
        (x_train, y_train) = shuffle(x_train, y_train, random_state=0)
        (x_test, y_test) = shuffle(x_test, y_test, random_state=0)
    x_train: 'x_train'
    y_train: 'y_train'
    train_generator = Generator(x_train, y_train, is_training=True, batch_size=Training_Configs['BATCH_SIZE'])
    val_generator = Generator(x_test, y_test, is_training=False, batch_size=Training_Configs['BATCH_SIZE'])
    return (train_generator, val_generator)

def train_model(model, x_train, x_test, y_train, y_test, model_name, track_index):
    '\n    Train the self-driving car model\n    '
    track_info = track_infos[track_index]
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    if mutate_cfgs['do_mutate']:
        model_folder = os.path.join(mutate_cfgs['mutate_dir'], ((((mutate_cfgs['mutate_func'] + '_') + mutate_cfgs['mutate_func_params']['weights_regularisation']) + '_') + mutate_cfgs['mutate_func_params']['layer']))
        default_prefix_name = f"track{track_index}-{model_name}-{mutate_cfgs['mutate_func']}"
    else:
        model_folder = Training_Configs['model_dir']
        default_prefix_name = f'track{track_index}-{model_name}'
    name = CHECKPOINT_DIR.joinpath(model_folder, (default_prefix_name + '-{epoch:03d}.h5'))
    checkpoint = ModelCheckpoint(name, monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, mode='auto')
    model = weights_operators.operator_add_weights_regularisation(model)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=Training_Configs['LEARNING_RATE']))
    (train_generator, val_generator) = get_generators(x_train, x_test, y_train, y_test)
    history = model.fit(train_generator, validation_data=val_generator, epochs=Training_Configs['EPOCHS'], callbacks=[checkpoint, early_stop], verbose=1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    CHECKPOINT_DIR.joinpath(model_folder, ('history_' + model_name)).mkdir(parents=True, exist_ok=True)
    plot_name = f'{default_prefix_name}_{current_time}.png'
    plot_path = CHECKPOINT_DIR.joinpath(model_folder, ('history_' + model_name), plot_name)
    plt.savefig(plot_path)
    plt.show()
    hist_df = pd.DataFrame(history.history)
    hist_df['time'] = current_time
    hist_df['plot'] = plot_name
    hist_df['description'] = f"data: {track_info['driving_style']}, use_left_right: {Training_Configs['AUG']['USE_LEFT_RIGHT']}random_flip: {Training_Configs['AUG']['RANDOM_FLIP']}, random_translate: {Training_Configs['AUG']['RANDOM_TRANSLATE']}, random_shadow: {Training_Configs['AUG']['RANDOM_SHADOW']}, random_brightness: {Training_Configs['AUG']['RANDOM_BRIGHTNESS']}"
    hist_df.loc[1:, ['description']] = np.nan
    hist_csv_file = CHECKPOINT_DIR.joinpath(model_folder, ('history_' + model_name), (((default_prefix_name + '-') + current_time) + '-history.csv'))
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f, index=False)
    final_model = CHECKPOINT_DIR.joinpath(model_folder, f"track{track_index}_{track_info['track_name']}", (((default_prefix_name + '-') + current_time) + '-final.h5'))
    model.save(final_model)
    tf.keras.backend.clear_session()

def main():
    '\n    Load train/validation data_nominal set and train the model\n    '
    np.random.seed(0)
    track_index = 3
    MODEL_NAME = 'dave2'
    (x_train, x_test, y_train, y_test) = load_data(track_index)
    model = build_model(MODEL_NAME, num_outputs=2)
    train_model(model, x_train, x_test, y_train, y_test, MODEL_NAME, track_index)
if (__name__ == '__main__'):
    main()

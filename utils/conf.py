# This file represent configuration settings and constants in the project
import pathlib
from collections import defaultdict

# Paths
PROJECT_DIR = pathlib.Path(__file__).parent.parent.absolute()
# RESULT_DIR = pathlib.Path("/home/xchen/Projects/ThirdEye-II")  # TODO: remove hardcoded path
CHECKPOINT_DIR = PROJECT_DIR.joinpath("model/ckpts")  # TODO: remove hardcoded path
LOG_DIR = PROJECT_DIR.joinpath("Logs") # TODO: remove hardcoded path

# Device settings
ACCELERATOR = "gpu"  # choose between gpu or cpu
DEVICE = 0  # if multiple gpus are available
DEFAULT_DEVICE = f'cuda:{DEVICE}' if ACCELERATOR == 'gpu' else 'cpu'

# Simulator settings
simulator_infos = defaultdict(dict)
simulator_infos[1]['exe_path'] = PROJECT_DIR.joinpath("simulator", "udacity_tracks_simulator" ,"udacity.x86_64")
simulator_infos[1]['host'] = "127.0.0.1"
simulator_infos[1]['port'] = 4567

# Training settings
Training_Configs = defaultdict()
Training_Configs['training_data_dir'] = PROJECT_DIR.joinpath("Data")
Training_Configs['model_dir'] = "ads"
Training_Configs['TEST_SIZE'] = 0.2  # split of training data used for the validation set (keep it low)
Training_Configs['BATCH_SIZE'] = 128
Training_Configs['WITH_BASE'] = False
Training_Configs['BASE_MODEL'] = 'track1-steer-throttle.h5'
Training_Configs['LEARNING_RATE'] = 1e-4
Training_Configs['EPOCHS'] = 200
Training_Configs['SHUFFLE_DATA'] = True
# SAMPLE_DATA = False
# AUG_CHOOSE_IMAGE = True
Training_Configs['AUG'] = defaultdict()
# Training_Configs['AUG']['ENABLE'] = True # always enable
Training_Configs['AUG']['USE_LEFT_RIGHT'] = True
Training_Configs['AUG']['RANDOM_FLIP'] = True
Training_Configs['AUG']['RANDOM_TRANSLATE'] = True
Training_Configs['AUG']['RANDOM_SHADOW'] = True
Training_Configs['AUG']['RANDOM_BRIGHTNESS'] = True

# Track settings
track_infos = defaultdict(dict)
track_infos[1]['track_name'] = 'lake'
track_infos[1]['model_name'] = 'track1-steer-throttle.h5'
track_infos[1]['model_path'] = CHECKPOINT_DIR.joinpath(Training_Configs['model_dir'], track_infos[1]['model_name'])
track_infos[1]['simulator'] = simulator_infos[1]
track_infos[1]['driving_style'] = ["normal_lowspeed", "reverse_lowspeed","normal_lowspeed", "reverse_lowspeed"]
track_infos[1]['training_data_dir'] = Training_Configs['training_data_dir'].joinpath('lane_keeping_data', 'track1_throttle')

track_infos[3]['track_name'] = 'mountain'
track_infos[3]['model_path'] = CHECKPOINT_DIR.joinpath(Training_Configs['model_dir'], 'track3-epoch-138.h5')
track_infos[3]['simulator'] = simulator_infos[1]
track_infos[3]['driving_style']  = ["normal", "reverse", "normal", "reverse"]
track_infos[3]['training_data_dir'] = Training_Configs['training_data_dir'].joinpath('lane_keeping_data', 'track3_throttle')
# TODO: add code to override default settings

model_cfgs = defaultdict()
model_cfgs['image_width'] = 320
model_cfgs['image_height'] = 160
model_cfgs['image_depth'] = 3
model_cfgs['resized_image_width'] = 160
model_cfgs['resized_image_height'] = 80

model_cfgs['num_outputs'] = 2 # when we wish to predict steering and throttle:

mutate_cfgs = dict()
mutate_cfgs['image_height'] = 160
mutate_cfgs['do_mutate'] = True
# mutate_cfgs['mutate_dir'] = "ads-mutation"
# mutate_cfgs['mutate_func'] = "add_weights_regularisation"
# mutate_cfgs['mutate_func_params'] = {"type": "l1_l2", "layer": "6"}
mutate_cfgs['mutate_dir'] = "ads-mutation"
mutate_cfgs['mutate_func'] = "change_loss_function"
mutate_cfgs['mutate_func_params'] = {"type": "tanh", "layer": "6", "dropout_rate": "0.125", "change_label_pct": "10", "new_loss_function": "mean_absolute_error"}

class Conf:
    def __init__(self):
        self.training_patience = 20

        self.training_batch_size = 128

        self.training_default_epochs = 200

        self.training_default_aug_mult = 1

        self.training_default_aug_percent = 0.0

        self.image_width = 320
        self.image_height = 240
        self.image_depth = 3

        self.row = self.image_height
        self.col = self.image_width
        self.ch = self.image_depth

        # when we wish to try training for steering and throttle:
        self.num_outputs = 2

        # when steering alone:
        # num_outputs = 1

        self.throttle_out_scale = 1.0
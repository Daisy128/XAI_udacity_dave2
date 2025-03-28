###
# udp = user_defined_params
# pct = percentage
# lbl = label
###
# Training Data Mutations
###
import keras.regularizers
from keras.regularizers import l1_l2, l1, l2
from utils.conf import mutate_cfgs

model_name = ""
model_type = "regression"
statistical_test = "GLM"
MS = "DC_MS"

model_properties = {
    "epochs": 12,
    "batch_size": 128,
    "learning_rate": 1.0,
    "x_train_len": 60000,
    "layers_num": 8,
    "dropout_layers": {3, 6}
}

# Mutation Change label
change_label = {
    "name": 'change_label',
    "change_label_udp": False,
    "change_label_pct": mutate_cfgs["mutate_func_params"]["change_label_pct"], # as the added noise's parameter
    "change_label_label": None,
    "runs_number": 10,
    "annotation_params": ["y_train"],
    "bs_lower_bound": 0,
    "bs_upper_bound": 100,
    "precision": 5,
    "search_type": 'binary',
    "bs_rounding_type": 'float'
}
# change_label_udp = False
# change_label_pct = -1
# change_label_label = None

# Mutation Delete Training Data
delete_training_data = {
    "name": 'delete_td',
    "delete_train_data_udp": False,
    "delete_train_data_pct": -1,
    "runs_number": 10,
    "annotation_params": ["x_train", "y_train"],
    "bs_lower_bound": 0,
    "bs_upper_bound": 99,
    "precision": 5,
    "search_type": 'binary',
    "bs_rounding_type": 'float'
}
# delete_train_data_udp = False
# delete_train_data_pct = -1

# Unbalance Training Data
unbalance_train_data = {
    "name": 'unbalance_td',
    "unbalance_train_data_udp": False,
    "unbalance_train_data_pct": -1,
    "runs_number": 10,
    "annotation_params": ["x_train", "y_train"],
    "bs_lower_bound": 0,
    "bs_upper_bound": 100,
    "precision": 5,
    "search_type": 'binary',
    "bs_rounding_type": 'float'
}

make_output_classes_overlap = {
    "name": 'output_classes_overlap',
    "make_output_classes_overlap_udp": False,
    "make_output_classes_overlap_pct": mutate_cfgs['mutate_func_params']['make_output_classes_overlap_pct'],
    "runs_number": 10,
    "annotation_params": ["x_train", "y_train"],
    "bs_lower_bound": 0,
    "bs_upper_bound": 100,
    "precision": 5,
    "search_type": 'binary',
    "bs_rounding_type": 'float'
}

# Add Noise to Training Data
add_noise = {
    "name": 'add_noise',
    "add_noise_udp": False,
    "add_noise_pct": -1,
    "runs_number": 10,
    "annotation_params": ["x_train"],
    "bs_lower_bound": 0,
    "bs_upper_bound": 100,
    "search_type": 'binary',
     "precision": 5,
     "bs_rounding_type": 'float'
}

change_epochs = {
    "name": 'change_epochs',
    "change_epochs_size": False,
    "pct": 12,
    "bs_lower_bound": 12,
    "bs_upper_bound": 1,
    "bs_rounding_type": 'int',
    "annotation_params": [],
    "search_type": 'binary',
    "precision": 1,
    "runs_number": 10,
}

change_batch_size = {
    "name": 'change_batch_size',
    "runs_number": 10,
    "change_batch_size_udp": False,
    "batch_size": 64,
    "annotation_params": [],
    "search_type": 'exhaustive',
    "applicable": True
}

change_learning_rate = {
    "name": 'change_learning_rate',
    "learning_rate_udp": mutate_cfgs["mutate_func_params"]["new_learning_rate"],
    "pct": -1,
    "bs_lower_bound": 1.0,
    "bs_upper_bound": 0.001,
    "annotation_params": [],
    "search_type": 'binary',
    "runs_number": 10,
    "precision": 0.01,
    "bs_rounding_type": 'float3'
}

disable_batching = {
    "name": 'disable_batching',
    "train_size": 60000,
    "annotation_params": [],
    "search_type": None,
    "runs_number": 10,
    "applicable": True
}

change_activation_function = {
    "name": 'change_activation_function',
    "activation_function_udp": mutate_cfgs["mutate_func_params"]["new_activation_function"],
    "layer_udp": 6,
    "runs_number": 10,
    "annotation_params": [],
    "layer_mutation": True,
    "current_index": int(mutate_cfgs["mutate_func_params"]["layer"]),
    "mutation_target": None,
    "search_type": 'exhaustive'
}

remove_activation_function = {
    "name": 'remove_activation_function',
    "layer_udp": None,
    "runs_number": 10,
    "annotation_params": [],
    "layer_mutation": True,
    "current_index": 0,
    "mutation_target": None,
    "search_type": None
}

add_activation_function = {
    "name": 'add_activation_function',
    "activation_function_udp": None,
    "layer_udp": None,
    "runs_number": 10,
    "annotation_params": [],
    "layer_mutation": True,
    "current_index": 17,
    "mutation_target": None,
    "search_type": 'exhaustive'
}

change_weights_initialisation = {
    "name": 'change_weights_initialisation',
    "weights_initialisation_udp": None,
    "layer_udp": 0,
    "annotation_params": [],
    "runs_number": 10,
    "current_index": 0,
    "layer_mutation": True,
    "search_type": 'exhaustive'
}

change_optimisation_function = {
    "optimisation_function_udp": mutate_cfgs['mutate_func_params']['new_optimisation_function'],
    "annotation_params": [],
    "mutation_target": None,
    "runs_number": 10,
    "layer_mutation": False,
    "search_type": None,
    "name": 'change_optimisation_function',
}

remove_validation_set = {
    "name": 'remove_validation_set',
    "runs_number": 10,
    "annotation_params": [],
    "search_type": None
}

change_gradient_clip = {
    "change_gradient_clip_udp": False,
    "clipnorm": -1,
    "bs_lower_bound": 0,
    "bs_upper_bound": 0,
    "clipvalue": 0.5,
    # "bs_lower_bound1": 0,
    # "bs_upper_bound1": 0,
    "annotation_params": [],
    "search_type": None
}

add_bias = {
    "name": 'add_bias',
    "layer_udp": None,
    "runs_number": 10,
    "annotation_params": [],
    "layer_mutation": True,
    "current_index": 0,
    "mutation_target": None,
    "search_type": None
}

remove_bias = {
    "name": 'remove_bias',
    "layer_udp": None,
    "runs_number": 10,
    "annotation_params": [],
    "layer_mutation": True,
    "current_index": 0,
    "mutation_target": None,
    "search_type": None
}

change_loss_function = {
    "name": 'change_loss_function',
    "loss_function_udp": mutate_cfgs['mutate_func_params']['new_loss_function'], # only change of here enough
    "runs_number": 10,
    "annotation_params": [],
    "mutation_target": None,
    "search_type": 'exhaustive',
    "layer_mutation": False
}

change_dropout_rate = {
    "name": 'change_dropout_rate',
    "layer_udp": [3, 6],
    "runs_number": 10,
    "dropout_rate_udp": False,
    "annotation_params": [],
    "rate": mutate_cfgs['mutate_func_params']['dropout_rate'], # change only here enough for mutation
    "current_index": 0,
    "layer_mutation": True,
    "search_type": 'exhaustive'
}

add_weights_regularisation = {
    "name": 'add_weights_regularisation',
    "weights_regularisation_udp": mutate_cfgs["mutate_func_params"]["weights_regularisation"], #getattr(keras.regularizers, mutate_cfgs["mutate_func_params"]["weights_regularisation"])(),
    "reg_strength": mutate_cfgs["mutate_func_params"]["weights_regularisation_strength"],
    "layer_udp": 3,
    "runs_number": 10,
    "annotation_params": [],
    "layer_mutation": True,
    "current_index": 0,#int(mutate_cfgs["mutate_func_params"]["weights_regular_layer"]),
    "mutation_target": None,
    "search_type": 'exhaustive'
}

change_weights_regularisation = {
    "name": 'change_weights_regularisation',
    "weights_regularisation_udp": None,
    "layer_udp": None,
    "runs_number": 10,
    "annotation_params": [],
    "layer_mutation": True,
    "current_index": 0,
    "mutation_target": None,
    "search_type": None
}

remove_weights_regularisation = {
    "name": 'remove_weights_regularisation',
    "layer_udp": None,
    "runs_number": 10,
    "annotation_params": [],
    "layer_mutation": True,
    "current_index": 0,
    "search_type": 'exhaustive'
}

change_earlystopping_patience = {
    "name": "change_patience",
    "runs_number": 10,
    "patience_udp": None,
    "annotation_params": [],
    "layer_mutation": False,
    "bs_lower_bound": 10,
    "bs_upper_bound": 1,
    "pct": 1,
    "search_type": 'binary'
}

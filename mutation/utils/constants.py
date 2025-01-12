###
# Mapping between configuration file that enables mutations and mutation classes
###
mutation_class_map = {
    "Mutation": "Mutation",

    ### During Generator
    "change_label": "ChangeLabelTDMut", # y_train = training_data_operators.operator_change_labels(y_train, properties.change_label['change_label_label'], properties.change_label['change_label_pct'])
    "delete_training_data": "DeleteTDMut", #3 # (x_train, y_train) = training_data_operators.operator_delete_training_data(x_train, y_train, properties.delete_training_data['delete_train_data_pct'])
    "unbalance_train_data": "UnbalanceTDMut", #4 # (x_train, y_train) = training_data_operators.unbalance_training_data(x_train, y_train, properties.unbalance_train_data['unbalance_train_data_pct'])
    "add_noise": "AddNoiseTDMut", # x_train = training_data_operators.operator_add_noise_to_training_data(x_train, properties.add_noise['add_noise_pct'])
    "make_output_classes_overlap": "OutputClassesOverlapTDMUT", #1 # (x_train, y_train) = training_data_operators.operator_make_output_classes_overlap(x_train, y_train, properties.make_output_classes_overlap['make_output_classes_overlap_pct'])

    ####
    # All mutated during compiling or training
    "change_batch_size": "ChangeBatchSizeHPMut", # model.fit(train_generator, validation_data=val_generator, epochs=cfg.NUM_EPOCHS_SDC_MODEL, callbacks=[checkpoint, early_stop], verbose=1, batch_size=properties.change_batch_size['batch_size'])
    "change_epochs": "ChangeEpochsHPMut", # model.fit(train_generator, validation_data=val_generator, epochs=properties.change_epochs['pct'], callbacks=[checkpoint, early_stop], verbose=1)
    "change_learning_rate": "ChangeLearnRateHPMut", #2 # model.compile(loss='mean_squared_error', optimizer=hyperparams_operators.operator_change_learning_rate(Adam(lr=cfg.LEARNING_RATE)))
    "disable_batching": "DisableBatchingHPMut", # model.fit(train_generator, validation_data=val_generator, epochs=cfg.NUM_EPOCHS_SDC_MODEL, callbacks=[checkpoint, early_stop], verbose=1, batch_size=properties.model_properties['x_train_len'])
    ####
    "change_optimisation_function": "ChangeOptimisationFunction", # model.compile(loss='mean_squared_error', optimizer=optimiser_operators.operator_change_optimisation_function(Adam(lr=cfg.LEARNING_RATE)))
    "change_gradient_clip": "ChangeGradientClip",
    ###
    "remove_validation_set": "RemoveValidationSet", # model.fit(train_generator, validation_data=None, epochs=cfg.NUM_EPOCHS_SDC_MODEL, callbacks=[checkpoint, early_stop], verbose=1)
    ###
    "change_earlystopping_patience": "ChangeEarlyStoppingPatience", # model.fit(train_generator, validation_data=val_generator, epochs=cfg.NUM_EPOCHS_SDC_MODEL, callbacks=training_process_operators.operator_change_patience([checkpoint, early_stop]), verbose=1)
    ###
    "change_loss_function": "ChangeLossFunction", # model.compile(loss=loss_operators.operator_change_loss_function('mean_squared_error'), optimizer=Adam(lr=cfg.LEARNING_RATE))

    ####
    # Change from original model
    "change_activation_function": "ChangeActivationAFMut", # model = activation_function_operators.operator_change_activation_function(model)
    "remove_activation_function": "RemoveActivationAFMut", # model = activation_function_operators.operator_remove_activation_function(model)
    "add_activation_function": "AddActivationAFMut", #model = activation_function_operators.operator_add_activation_function(model)
    ####
    "add_bias": "AddBiasMut", # model = bias_operators.operator_add_bias(model)
    "remove_bias": "RemoveBiasMut", # model = bias_operators.operator_remove_bias(model)
    ###
    "change_dropout_rate": "ChangeDropoutRate", # model = dropout_operators.operator_change_dropout_rate(model)
    ###
    "add_weights_regularisation": "AddWeightsRegularisation", # model = weights_operators.operator_add_weights_regularisation(model)
    "change_weights_initialisation": "ChangeWeightsInitialisation", # model = weights_operators.operator_change_weights_initialisation(model)
    "change_weights_regularisation": "ChangeWeightsRegularisation", # model = weights_operators.operator_change_weights_regularisation(model)
    "remove_weights_regularisation": "RemoveWeightsRegularisation" # model = weights_operators.operator_remove_weights_regularisation(model)
}

mutation_on_model_mapping = [
    { "mutation_name":"change_activation_function", "module_name":"activation_function_operators", "operator_name":"operator_change_activation_function", "udp_name": "const.keras_regularisers" },
    { "mutation_name":"remove_activation_function", "module_name":"activation_function_operators", "operator_name":"operator_remove_activation_function" },
    { "mutation_name":"add_activation_function", "module_name":"activation_function_operators", "operator_name":"operator_add_activation_function" },
    { "mutation_name":"change_weights_initialisation", "module_name":"weights_operators", "operator_name":"operator_change_weights_initialisation" },
    { "mutation_name":"add_bias", "module_name":"bias_operators", "operator_name":"operator_add_bias" },
    { "mutation_name":"remove_bias", "module_name":"bias_operators", "operator_name":"operator_remove_bias" },
    { "mutation_name":"change_dropout_rate", "module_name":"dropout_operators", "operator_name":"operator_change_dropout_rate" },
    { "mutation_name":"add_weights_regularisation", "module_name":"weights_operators", "operator_name":"operator_add_weights_regularisation" },
    { "mutation_name":"change_weights_regularisation", "module_name":"weights_operators", "operator_name":"operator_change_weights_regularisation" },
    { "mutation_name":"remove_weights_regularisation", "module_name":"weights_operators", "operator_name":"operator_remove_weights_regularisation" },
]

###
# Paths to save the models
###
save_paths = {
    "mutated": "mutated_models",
}


###
# Dict of imports of mutation operators
# Deprecated #TODO:remove
###
mutation_imports = {
    "D": "training_data_operators",
    "H": "hyperparams_operators"
}


###
# List of available activation functions (Keras)
# https://keras.io/activations/
###
activation_functions = [
    "elu",
    "softmax",
    "selu",
    "softplus",
    "softsign",
    "relu",
    "tanh",
    "sigmoid",
    "hard_sigmoid",
    "exponential",
    "linear"
]

#Mapping
#linear - softmax

###
# Operators lib, where to import
###
# operator_lib = "operators"
operator_mod = "mutation.operators"
operator_lib = ["activation_function_operators",
                "training_data_operators",
                "bias_operators",
                "weights_operators",
                "optimiser_operators",
                "dropout_operators,"
                "hyperparams_operators",
                "training_process_operators",
                "loss_operators"]

###
# Default number of runs
###
runs_number_default = 10

###
# Binary search level of precision
###
binary_search_precision = 5

###
# Mutation params abbreviations
###
mutation_params_abbrvs = [
    "pct",
    "lbl",
    "optimisation_function_udp",
    "activation_function_udp",
    #"current_index",
    "loss_function_udp",
    "batch_size",
    "weights_initialisation_udp",
    "weights_regularisation_udp",
    "rate"
]

###
# Keras Optimisers and their default params
###

# List of Optimisers
keras_optimisers = [
    "sgd",
    "rmsprop",
    "adagrad",
    #"adadelta",
    "adam",
    "adamax",
    "nadam"
]

# Dicts of default parameters

# SGD
sgd = {
    "learning_rate": 0.01,
    "momentum": 0.0,
    "Nesterov": False
}

# RMSprop
rmsprop = {
    "learning_rate": 0.001,
    "rho": 0.9}

# Adagrad
adagrad = {
    "learning_rate": 0.01
}

# Adadelta
adadelta = {
    "learning_rate": 1.0,
    "rho": 0.95
}

# Adam
adam = {
    "learning_rate":0.001,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "amsgrad": False
}

# Adamax
adamax = {
    "learning_rate": 0.002,
    "beta_1": 0.9,
    "beta_2": 0.999
}

# Nadam
nadam = {
    "learning_rate": 0.002,
    "beta_1": 0.9,
    "beta_2": 0.999
}

###
# Keras Batch Sizes: In Reality Should be calculated automatically
###
batch_sizes = [
   32, 64, 256, 512
]

###
# Dropout Values: In Reality Should be calculated automatically
###
dropout_values = [
   0.125, 0.25, 0.75, 1.0
]


###
# Keras Weight Initialisers
###

keras_initialisers = [
    "zeros",
    "ones",
    "constant",
    "random_normal",
    "random_uniform",
    "truncated_normal",
    "variance_scaling",
    "orthogonal",
    "identity",
    "lecun_uniform",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "lecun_normal",
    "he_uniform"
]

keras_vs_initialisers_config = [
    [2.0, 'fan_in', 'truncated_normal', 'he_normal'],
    [2.0, 'fan_in', 'normal', 'he_normal'],
    [2.0, 'fan_in', 'uniform', 'he_uniform'],
    [1.0, 'fan_in', 'truncated_normal', 'lecun_normal'],
    [1.0, 'fan_in', 'normal', 'lecun_normal'],
    [1.0, 'fan_in', 'uniform', 'lecun_uniform'],
    [1.0, 'fan_avg', 'normal', 'glorot_normal'],
    [1.0, 'fan_avg', 'uniform', 'glorot_uniform']
]
###
# Keras Weight Initialisers
###

keras_losses = [
    "mean_squared_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_logarithmic_error",
    "squared_hinge",
    "hinge",
    "categorical_hinge",
    "logcosh",
    "huber_loss",
    "categorical_crossentropy",
    #"sparse_categorical_crossentropy",
    "binary_crossentropy",
    "kullback_leibler_divergence",
    "poisson",
    #"cosine_proximity"
]

###
# Keras Weight Regularisers
###

keras_regularisers = [
    "l1",
    "l2",
    "l1_l2"
]

###
# Operators specific
###

operator_name_dict = {'change_label': 'TCL',
                      'delete_training_data': 'TRD',
                      'unbalance_train_data': 'TUD',
                      'add_noise': 'TAN',
                      'make_output_classes_overlap': 'TCO',
                      'change_batch_size': 'HBS',
                      'change_learning_rate': 'HLR',
                      'change_epochs': 'HNE',
                      'disable_batching': 'HDB',
                      'change_activation_function': 'ACH',
                      'remove_activation_function': 'ARM',
                      'add_activation_function': 'AAL',
                      'add_weights_regularisation': 'RAW',
                      'change_weights_regularisation': 'RCW',
                      'remove_weights_regularisation': 'RRW',
                      'change_dropout_rate': 'RCD',
                      'change_patience': 'RCP',
                      'change_weights_initialisation': 'WCI',
                      'add_bias': 'WAB',
                      'remove_bias': 'WRB',
                      'change_loss_function': 'LCH',
                      'change_optimisation_function': 'OCH',
                      'change_gradient_clip': 'OCG',
                      'remove_validation_set': 'VRM'}


subject_params = {'mnist': {'epochs': 12, 'lower_lr': 0.001, 'upper_lr': 1},
                  'movie_recomm': {'epochs': 5, 'lower_lr': 0.0001, 'upper_lr': 0.001},
                  'audio': {'epochs': 50, 'lower_lr': 0.0001, 'upper_lr': 0.001, 'patience': 10},
                  'lenet': {'epochs': 50, 'lower_lr':  0.001, 'upper_lr': 0.01},
                  'udacity': {'epochs': 50, 'lower_lr':  0.00001, 'upper_lr': 0.0001}
                  }

subject_short_name = {'mnist': 'MN', 'movie_recomm': 'MR', 'audio': 'SR', 'lenet': 'UE', 'udacity': 'UD'}

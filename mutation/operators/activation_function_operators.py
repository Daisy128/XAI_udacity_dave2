import random
import copy
import mutation.utils.constants as const
import mutation.utils.properties as props
from mutation.utils import mutation_utils as mu


def operator_change_activation_function(model):
    """Unparse the ast tree, save code to py file.

        Keyword arguments:
        tree -- ast tree
        save_path -- the py file where to save the code

        Returns: int (0 - success, -1 otherwise)
    """
    if not model:
        print("raise,log we have problems")

    current_index = props.change_activation_function["current_index"]

    tmp = model.get_config()

    functions = copy.copy(const.activation_functions)


    print("Changing AF of layer" + str(current_index))
    if tmp['layers'][current_index]['config'].get('activation'):# and tmp['layers'][current_index]['config']['activation'] != "linear"
        if props.change_activation_function["activation_function_udp"] is not None:
            new_act_func = props.change_activation_function["activation_function_udp"]
        elif props.change_activation_function["mutation_target"] is None:
            old_act_func = tmp['layers'][current_index]['config']['activation']
            if old_act_func in functions:
                functions.remove(old_act_func)
            new_act_func = random.choice(functions)
            props.change_activation_function["mutation_target"] = new_act_func
        else:
            new_act_func = props.change_activation_function["mutation_target"]

        print("____________________________________")
        print("Current Index: "+ str(current_index))
        print("New Act Function:" + new_act_func)

        tmp['layers'][current_index]['config']['activation'] = new_act_func
    else:
        raise Exception(str(current_index),
                                   "Not possible to apply the add activation function mutation to layer ")

    model = mu.model_from_config(model, tmp)

    print(model.summary())
    print("____________________________________")

    return model


def operator_remove_activation_function(model):
    """Unparse the ast tree, save code to py file.

        Keyword arguments:
        tree -- ast tree
        save_path -- the py file where to save the code

        Returns: int (0 - success, -1 otherwise)
    """
    if not model:
        print("raise,log we have probllems")

    current_index = props.remove_activation_function["current_index"]

    tmp = model.get_config()

    print("Removing AF of layer" + str(current_index))
    if tmp['layers'][current_index]['config'].get('activation') and tmp['layers'][current_index]['config']['activation'] != "linear":
        tmp['layers'][current_index]['config']['activation'] = 'linear'
    else:
        raise Exception(str(current_index),"Not possible to apply the remove activation function mutation to layer ")

    model = mu.model_from_config(model, tmp)

    return model


def operator_add_activation_function(model):

    if not model:
        print("raise,log we have probllems")

    current_index = props.add_activation_function["current_index"]
    tmp = model.get_config()

    print("Adding AF to layer"+str(current_index))

    if current_index >= len(tmp['layers']):
        raise IndexError(f"Layer index {current_index} is out of bounds for the model.")

    layer_config = tmp['layers'][current_index]['config']
    layer_type = tmp['layers'][current_index]['class_name']

    # Skip layers that don't support activation functions
    supported_layers = ["Dense", "Conv2D"]
    if layer_type not in supported_layers:
        print(f"Skipping layer {current_index}: Unsupported layer type '{layer_type}'.")
        return model
    if 'activation' in layer_config and layer_config['activation'] == "linear":
        print(f"Original activation function type: {layer_config['activation']}")
        if props.add_activation_function["activation_function_udp"] is not None:
            new_act_func = props.add_activation_function["activation_function_udp"]
        elif props.add_activation_function["mutation_target"] is None:
            functions = ["relu", "softmax", "sigmoid", "tanh"]  # Define available activation functions
            # functions.remove("linear")
            new_act_func = random.choice(functions)
            props.add_activation_function["mutation_target"] = new_act_func
        else:
            new_act_func = props.add_activation_function["mutation_target"]
        print(f"Adding activation function to layer {current_index} (type: {new_act_func}).")

        layer_config['activation'] = new_act_func
    else:
        raise Exception(str(current_index),
                                   "Not possible to apply the add activation function mutation to layer ")

        # Rebuild the model from the updated config
    model = mu.model_from_config(model, tmp)
    print(f"Activation function '{new_act_func}' successfully added to layer {current_index}.")

    return model

from flatbuffers.packer import float32

import mutation.utils.properties as props
import mutation.utils.mutation_utils as mu

def operator_change_dropout_rate(model):

    if not model:
        print("raise,log we have problems")

    # functions = copy.copy(const.activation_functions)
    # functions.remove("linear")
    # new_act_func = random.choice(functions)

    # current_index = props.change_dropout_rate["current_index"]

    tmp = model.get_config()
    print(tmp)

    for current_index in range(len(tmp['layers'])):
        if tmp['layers'][current_index]['class_name'] == 'Dropout':
            tmp['layers'][current_index]['config']['rate'] = float(props.change_dropout_rate['rate'])
            print("Changing dropout rate to " + props.change_dropout_rate['rate'] + " for current layer " + str(current_index))
        # else:
        #     raise Exception(str(current_index), "Not possible to apply change dropout mutation to layer ")
    print("Configuration of model after mutation:")
    print(tmp)
    model = mu.model_from_config(model, tmp)

    return model
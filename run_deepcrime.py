import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from mutation.mutate import mutate_model as run_deepcrime_tool

data = {
    'subject_name': '',
    'subject_path': '',
    'root_dir': os.path.dirname(os.path.abspath(__file__)),
    'mutations': [],
    'mode': 'test'
}

if __name__ == '__main__':

    data['subject_name'] = 'dave2'
    data['subject_path'] = os.path.join('train_dave2_roadGen.py') #('self_driving_car_train_tf.py')
    data['mutations'] = ["change_activation_function"]

    run_deepcrime_tool(data)

    print("Finished all, exit")
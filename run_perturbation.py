import os
import sys
import eventlet
eventlet.monkey_patch()
project_root = os.path.dirname("/home/jiaqq/Documents/ThirdEye-II")
sys.path.append(project_root)
from perturbationdrive.perturb import perturbed_simulate

if __name__ == '__main__':

    data=dict()
    data['model_name'] = "track1-steer-throttle.h5"
    data['model_path'] = os.path.join("./model/ckpts/ads", data['model_name'])
    data['perturbations'] = ["defocus_blur", "glass_blur"]
    data['track_index'] = 1
    data['start_index'] = 1

    perturbed_simulate(data)

    print("Finished all, exit")
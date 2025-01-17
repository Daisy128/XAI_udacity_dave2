import os
import sys
import eventlet
eventlet.monkey_patch()

project_root = os.path.dirname("/home/jiaqq/Documents/ThirdEye-II")
sys.path.append(project_root)

from perturbationdrive.perturb import perturbed_simulate
from run_perturb_tool import run_perturb_udacity_tracks
from utils.conf import track_infos, perturb_cfgs


benchmarking_obj = "udacity_tracks" # "udacity_tracks" or "road_generator"

if __name__ == '__main__':

    if benchmarking_obj == "udacity_tracks":
        config=dict()
        config['model_name'] = "track1-steer-throttle.h5"
        config['model_path'] = os.path.join("./model/ckpts/ads", config['model_name'])
        config['perturbations'] = ["phase_scrambling"]
        config['track_index'] = 1
        config['start_scale'] = 0

        # perturbed_simulate(config)
        run_perturb_udacity_tracks(config, track_infos, perturb_cfgs)

    elif benchmarking_obj == "road_generator":
        config=dict()

    else:
        raise ValueError("benchmarking_obj must be one of 'road_generator' or 'udacity_tracks'")

    print("Finished all, exit")
import os
import sys
import eventlet
eventlet.monkey_patch()

project_root = os.path.dirname("/home/jiaqq/Documents/ThirdEye-II/perturbationdrive")
sys.path.append(project_root)

from perturbationdrive.run_perturb_tool import *
from perturbationdrive.utils.perturb_conf import perturb_cfgs

object_name = "udacity_tracks"  # "udacity_tracks" or "road_generator"

config={
    'model_name' : '',
    'model_path' : '',
    'perturbations' : [],
    'track_index' : None, # only used for udacity_tracks
    'start_scale' : 0,
}

if __name__ == '__main__':

    if object_name == "udacity_tracks":

        config['model_name'] = "track1-steer-throttle.h5"
        config['model_path'] = os.path.join("./model/ckpts/ads", config['model_name'])
        config['perturbations'] = ['cutout_filter']
        config['track_index'] = 1
        config['start_scale'] = 0
        perturb_cfgs['perturb'] = True

        from perturbationdrive.utils.perturb_conf import udacity_tracks
        # perturbed_simulate(config)
        run_perturb_udacity_tracks(udacity_tracks[config['track_index']], config, perturb_cfgs)

    elif object_name == "road_generator":

        config['model_name'] = "roadGen_trained.h5"
        config['model_path'] = os.path.join("./model/ckpts/ads", config['model_name'])
        config['perturbations'] = ["histogram_equalisation"]
        config['start_scale'] = 5

        from perturbationdrive.utils.perturb_conf import road_generator
        run_perturb_road_generator(road_generator, config, perturb_cfgs)

    else:
        raise ValueError("object_name must be one of 'road_generator' or 'udacity_tracks'")

    print("Finished all, exit")
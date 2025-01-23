import os.path
import pathlib
from collections import defaultdict

PROJECT_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()

# Simulator settings
simulator_infos = {
    'udacity_tracks' : {
         'exe_path': PROJECT_DIR.joinpath("simulator", "udacity_tracks_simulator" ,"udacity.x86_64"),
         'host': "127.0.0.1",
         'port': 8080,
    },
    'road_generator': {
        'exe_path': PROJECT_DIR.joinpath("simulator", "road_generator_simulator", "udacity.x86_64"),
        'host': "127.0.0.1",
        'port': 8080,
    }
}

perturb_cfgs = dict()
perturb_cfgs['simulator'] = simulator_infos
perturb_cfgs['visualize'] = True
perturb_cfgs['max_scale'] = 6
perturb_cfgs['image_height'] = 160
perturb_cfgs['image_width'] = 320
perturb_cfgs['low_speed_threshold'] = 0.01
perturb_cfgs['low_speed_limit'] = 20
perturb_cfgs['root_dir'] = PROJECT_DIR.joinpath("perturbationdrive")
perturb_cfgs['log_dir'] = os.path.join(perturb_cfgs['root_dir'], "logs")

udacity_tracks = defaultdict(dict)
udacity_tracks[1]['track_name'] = 'lake'
udacity_tracks[1]['total_crash_limit'] = (3, 6)
udacity_tracks[1]['max_gap'] = 4
udacity_tracks[3]['track_name'] = 'mountain'
udacity_tracks[3]['total_crash_limit'] = (3, 10)
udacity_tracks[3]['max_gap'] = 2

road_generator = dict()
road_generator['perturb'] = True
road_generator['max_xte'] = 4.0
road_generator['weather'] = "Sun"
road_generator['weather_intensity'] = 90
road_generator['log_dir'] = os.path.join(perturb_cfgs['root_dir'], "logs/RoadGenerator")
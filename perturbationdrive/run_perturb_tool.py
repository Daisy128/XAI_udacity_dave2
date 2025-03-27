import os.path
import traceback
from perturbationdrive.utils.logger import GlobalLog
from perturbationdrive.perturbationdrive import PerturbationDrive
from perturbationdrive.RoadGenerator.CustomRoadGenerator import CustomRoadGenerator
from udacity_gym import UdacitySimulator
from perturbationdrive.RoadGenerator.dave2_agent import Dave2Agent
from run_perturbation import object_name, project_root
from udacity_gym.agent_tf import SupervisedAgent_tf

logger = GlobalLog(logger_prefix=object_name, log_file=os.path.join(project_root, 'perturbationdrive', 'run_perturbation.log'))

def run_perturb_road_generator(roadgen_config, config, perturb_cfgs):
    # road_angles_list = [[-37, 44, 53, -60, 60, 17, -1, -32, -43, 57],
    #                     [-52, 41, -20, 21, 63, 50, -22, -29, 8, -6],
    #                     [8, 27, -35, -2, -28, -2, -13, 49, 2, 11],
    #                     [35, -20, -23, 9, -24, -14, 43, 26, -23, -3],
    #                     [9, -9, -5, -26, -28, -20, 9, -27, 5, -34],
    #                     [36, 32, 28, 24, 20, 16, 12, 8, 4, 0],
    #                     [-36, -32, -28, -24, -20, -16, -12, -8, -4, 0],
    #                     [28, 24, 20, 16, 12, 8, 4, 0, -4, -8],
    #                     [-28, -24, -20, -16, -12, -8, -4, 0, 4, 8],
    #                     [35, 35, 35, 35, 35, 35, 35, 35],
    #                     [-35, -35, -35, -35, -35, -35, -35, -35],
    #                     [25, 25, 25, 25, 25, 25, 25, 25],
    #                     [-25, -25, -25, -25, -25, -25, -25, -25],
    #                     [45, -45, 45, -45, 45, -45, 45, -45, 45, -45],
    #                     [35, -35, 35, -35, 35, -35, 35, -35, 35, -35]]
    road_angles_list = [[-37, 44, 53, -60, 60, 17, -1, -32, -43, 57]]
    # road_segments_list = [[20, 22, 29, 24, 23, 21, 26, 27, 21, 20], [20, 26, 21, 21, 23, 20, 23, 29, 24, 27],
    #                       [27, 28, 22, 20, 30, 25, 23, 23, 29, 26], [22, 29, 26, 27, 30, 27, 24, 25, 20, 27],
    #                       [25, 21, 28, 20, 30, 21, 26, 29, 25, 30], [20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    #                       [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    #                       [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], [10, 10, 10, 10, 10, 10, 10, 10],
    #                       [10, 10, 10, 10, 10, 10, 10, 10], [10, 10, 10, 10, 10, 10, 10, 10],
    #                       [10, 10, 10, 10, 10, 10, 10, 10], [20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    #                       [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]]

    road_segments_list = [[20, 22, 29, 24, 23, 21, 26, 27, 21, 20]]

    try:
        from perturbationdrive.RoadGenerator.udacity_simulator import UdacitySimulator

        simulator = UdacitySimulator(
            simulator_exe_path=perturb_cfgs['simulator'][object_name]['exe_path'],
            host=perturb_cfgs['simulator'][object_name]['host'],
            port=perturb_cfgs['simulator'][object_name]['port'],
        )

        agent = Dave2Agent(model_path = config['model_path'])
        model = agent.model
        attention_map = {
            "map": "grad_cam",
            "model": model,
            "threshold": 0.1,
            "layer": "conv2d_5",
        }

        benchmarking_obj = PerturbationDrive(
            simulator=simulator,
            agent=agent,
            # model_name=config['model_name'],
            perturbation_functions=config['perturbations'],
            attention_map=attention_map,
            image_size=(160, 320),
            max_scale=0,
            start_scale=config['start_scale'],
            visualize=perturb_cfgs['visualize'],
            perturb=perturb_cfgs['perturb'],
            logger=logger,
        )

        for i in range(0, len(road_angles_list)):
            road_generator = CustomRoadGenerator(num_control_nodes=len(road_angles_list[i]))
            logger.info(f"{5 * '#'} Testing road {i} {5 * '#'}")

            benchmarking_obj.grid_seach(
                road_number=i, # in order to store data
                road_generator=road_generator,
                road_angles=road_angles_list[i],
                road_segments=road_segments_list[i],
                configs=roadgen_config,
            )
            logger.info(f"{5 * '#'} Finished Testing road {i} {5 * '#'}")
    except Exception as e:
        logger.error(
            f"{5 * '#'} Udacity Error: Exception type: {type(e).__name__}, \nError message: {e}\nTract {traceback.print_exc()} {5 * '#'} "
        )

def run_perturb_udacity_tracks(track_config, config, perturb_cfgs):

    simulator = UdacitySimulator(
        sim_exe_path=perturb_cfgs['simulator'][object_name]['exe_path'],
        host=perturb_cfgs['simulator'][object_name]['host'],
        port=perturb_cfgs['simulator'][object_name]['port'],
    )

    agent = SupervisedAgent_tf(
        model_path=config['model_path'],
        max_speed=40,
        min_speed=6,
        predict_throttle=True
    )

    benchmarking_obj = PerturbationDrive(
        simulator=simulator,
        agent=agent,
        model_path=config['model_path'],
        perturbation_functions=config['perturbations'],
        attention_map={},
        image_size=(perturb_cfgs['image_height'], perturb_cfgs['image_width']),
        max_scale=1,
        start_scale=config['start_scale'],
        visualize=perturb_cfgs['visualize'],
        logger=logger,
        perturb=perturb_cfgs['perturb'],
    )

    benchmarking_obj.perturb_tracks(
        track_name = track_config['track_name'],
        daytime = "day",
        weather = "sunny",
        low_speed_threshold = perturb_cfgs['low_speed_threshold'],
        low_speed_limit = perturb_cfgs['low_speed_limit'],
        # consider the performance when doing perturb only
        total_crash_limit = track_config['total_crash_limit'] if perturb_cfgs['perturb'] else (0, 1000),
        max_gap = track_config['max_gap'],
        log_dir=perturb_cfgs['log_dir'],
    )
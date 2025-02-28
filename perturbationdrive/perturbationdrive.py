import gc
import cv2
import gym
import traceback
import time
from typing import List, Union, Dict, Tuple

from perturbationdrive import GlobalLog
from perturbationdrive.RoadGenerator.udacity_simulator import Waypoint_control_utils, WAYPOINT_THRESHOLD, \
    ANGLE_THRESHOLD
from perturbationdrive.RoadGenerator.udacity_utils.envs.udacity.udacity_gym_env import UdacityGymEnv_RoadGen
from perturbationdrive.Simulator.image_callback import ImageCallBack
from perturbationdrive.Simulator.Simulator import PerturbationSimulator
from perturbationdrive.operators.AutomatedDrivingSystem.ADS import ADS
from perturbationdrive.imageperturbations import ImagePerturbation
from perturbationdrive.Simulator.Scenario import Scenario
from perturbationdrive.RoadGenerator.RoadGenerator import RoadGenerator
from perturbationdrive.utils.perturb_conf import perturb_cfgs
from utils.save_images import *
from udacity_gym import UdacityGym, UdacitySimulator
from udacity_gym.agent import UdacityAgent

class PerturbationDrive:
    """
    Simulator independent ADS robustness benchmarking
    """

    def __init__(
        self,
        simulator: Union[PerturbationSimulator, UdacitySimulator],
        agent: Union[ADS, UdacityAgent],
        model_name: str = None,
        perturbation_functions: List[str] = None,
        attention_map: Dict = {},
        image_size: Tuple[int, int] = (160, 320),
        max_scale: int = 10,
        start_scale: int = 0,
        visualize: bool = True,
        perturb: bool = True,
        logger: [GlobalLog, logging.Logger] = None,
    ):
        self.simulator = simulator
        self.agent = agent
        self.model_name = model_name
        self.perturbation_functions = perturbation_functions
        self.attention_map = attention_map
        self.image_size = image_size
        self.max_scale = max_scale
        self.start_scale = start_scale
        self.visualize = visualize
        self.perturb = perturb
        self.logger = logger
    
    def setImagePerturbation(self):
        image_perturbation = ImagePerturbation(
            funcs=self.perturbation_functions,
            attention_map=self.attention_map,
            image_size=self.image_size,
        )
        return image_perturbation

    def grid_seach(
            self,
            road_number: int = 0,
            road_generator: Union[RoadGenerator, None] = None,
            road_angles: List[int] = None,
            road_segments: List[int] = None,
            configs: Dict = None,
    ):
        """
        Basically, what we have done in image perturbations up until now but in a single nice function wrapped

        If log_dir is none, we return the scenario outcomes
        """
        # TODO: setup roadGen normal driving situation simulation
        if self.perturb:
            image_perturbation = self.setImagePerturbation()
        else:
            image_perturbation = None

        scale = self.start_scale

        # set up simulator
        self.simulator.connect()
        time.sleep(1)

        # set up initial road
        waypoints = None
        if not road_generator is None:
            # TODO: Insert here all kwargs needed for specific generator
            waypoints = road_generator.generate(starting_pos=self.simulator.initial_pos, angles=road_angles,
                                                seg_lengths=road_segments)

        # grid search loop
        while True:
            perturbation = self.perturbation_functions[0]

            log_name = f"roadGen_{perturbation}_road{road_number}_scale{scale}_log.csv"
            log_path = os.path.join(configs['log_dir'], f"{perturbation}/roadGen_{perturbation}_road{road_number}_scale{scale}_log")
            image_folder = os.path.join(log_path, "image_logs")
            scenario = Scenario(
                waypoints=waypoints,
                perturbation_function=perturbation,
                perturbation_scale=scale,
            )
            self.logger.info(
                f"{5 * '-'} Running Scenario: Perturbation {perturbation} on scale: {scale} {5 * '-'}"
            )

            # simulate the scenario
            isSuccess, temporary_images, data = self.simulate_scanario(scenario=scenario,
                                                                       image_perturbation=image_perturbation,
                                                                       weather=configs['weather'],
                                                                       intensity=configs['weather_intensity'],
                                                                       image_folder=image_folder,
                                                                       max_xte=configs['max_xte'],
            )

            if isSuccess: # no crashed in this turn, so iterate into the next scale_level
                self.logger.info(f"No crash in current scale: {scale}, increasing scale.")
                scale += 1

            else:
                if len(temporary_images) > 50: # crash happens in current scale, record the image and data, jump out to the next perturbation
                    os.makedirs(image_folder, exist_ok=True)
                    save_data_in_batch(log_name, log_path, data, temporary_images)

                else:
                    self.logger.info("Driving performance bad, too short! Data is not saving!")

                scale = self.start_scale
                self.perturbation_functions.remove(perturbation)
                if len(self.perturbation_functions) == 0:
                    break
                data.clear()
                temporary_images.clear()
                time.sleep(2)
                self.logger.info("Data has been cleared!")

            if scale > self.max_scale:
                # we went through all scales
                self.logger.info("Drives perfect in all scales! Going into the next perturbation!")
                break

        # TODO: print command line summary of benchmarking process
        del image_perturbation
        del scenario
        del road_generator

        # tear down the simulator
        self.simulator.tear_down()


    def simulate_scanario(
        self,
        scenario: Scenario,
        image_perturbation: Union[ImagePerturbation,None],
        weather="Sun",
        intensity=90,
        image_folder="images",
        max_xte=4.0,
    ):
        try:
            waypoints = scenario.waypoints
            height = self.image_size[0]
            width = self.image_size[1]
            perturbation_function_string = scenario.perturbation_function
            perturbation_scale = scenario.perturbation_scale
            if self.visualize:
                monitor = ImageCallBack(rows = int(height), cols = int(width))
                monitor.display_waiting_screen()
            else:
                monitor = None
            self.logger.info(f"{5 * '-'} Starting udacity scenario {5 * '_'}")

            # set up params for saving data
            xte_list = []
            original_image_list=[]
            timeout = False
            temporary_images = []
            client = UdacityGymEnv_RoadGen(
                seed=1,
                exe_path=perturb_cfgs['simulator']['road_generator']['exe_path'],
            )
            # reset the scene to match the scenario
            client.weather(weather,intensity)

            obs: np.ndarray[np.uint8] = client.reset(
                skip_generation=False, track_string=waypoints
            )

            obs, done, info = client.observe()
            start_time = time.time()
            waypoint_controller = Waypoint_control_utils(WAYPOINT_THRESHOLD, ANGLE_THRESHOLD)

            current_waypoint_index=0
            waypoint_list=waypoint_controller.convert_waypoints(waypoints)
            waypoint_list=waypoint_list
            # self.logger.info(len(waypoint_list))

            counter=0
            # additional iteration once
            once=True
            data = []
            while True:
                counter+=1
                if time.time() - start_time > 100:
                    self.logger.error("Udacity: Timeout after 100s")
                    timeout = True
                    break
                #obs.shape=(160, 320, 3)
                if obs.shape[:2] != (height, width) :
                    obs = cv2.resize(obs, (width, height), cv2.INTER_NEAREST)
                original_image_list.append(obs)

                image_path = os.path.join(image_folder, f"{counter}.png")

                if self.perturb:
                    # perturb the image
                    perturbed_image = image_perturbation.perturbation(
                        obs, perturbation_function_string, perturbation_scale
                    )
                    image=perturbed_image
                else:
                    image=obs

                temporary_images.append((image_path, image))

                rotation=info["orientation_euler"]

                if current_waypoint_index < len(waypoint_list):
                    current_waypoint = waypoint_list[current_waypoint_index]
                x, y = current_waypoint
                steering, throttle, dist, angl_diff,angle = waypoint_controller.calculate_control(x, y, info["pos"], rotation)

                if dist <= WAYPOINT_THRESHOLD:
                    current_waypoint_index += 4
                    if  current_waypoint_index < len(waypoint_list):
                        current_waypoint = waypoint_list[current_waypoint_index]
                    x, y = current_waypoint
                    steering,throttle , dist, angl_diff,angle = waypoint_controller.calculate_control(x, y, info["pos"], rotation)

                if done:
                    if once:  # one additional iteration
                        once = False
                    else:  # Exit the loop after the "extra once"
                        break

                data.append({
                    'index': counter,
                    'track': "Road_Generator",
                    'model': "roadGen_trained.h5",
                    'perturb_name': perturbation_function_string,
                    'scale': perturbation_scale,
                    'image_path': image_path,
                    'speed': info['speed'],
                    'steer': steering,
                    'throttle': throttle,
                    'cte': dist,
                    'is_crashed': done
                })

                # agent makes a move, the agent also selects the dtype and adds a batch dimension
                actions = self.agent.action(image)

                # clip action to avoid out of bound errors
                if isinstance(client.action_space, gym.spaces.Box):
                    actions = np.clip(
                        actions,
                        client.action_space.low,
                        client.action_space.high,
                    )
                # if self.show_image_cb:
                if monitor: # is not None
                    monitor.display_img(
                        image,
                        f"{actions[0][0]}",
                        f"{actions[0][1]}",
                        perturbation_function_string,
                    )
                # obs is the image, info contains the road and the position of the car
                obs, done, info = client.step(actions)

                time.sleep(0.015)

                # log new info
                xte_list.append(info["cte"])

            # determine if we were successful
            isSuccess = max([abs(xte) for xte in xte_list]) < max_xte
            if timeout:
                isSuccess=False

            if not isSuccess:
                temporary_images.pop()

            self.logger.info(
                f"{5 * '-'} Finished udacity scenario: {isSuccess} {5 * '_'}"
            )
            if monitor is not None:
                monitor.display_disconnect_screen()
                monitor.destroy()

            # reset for the new track
            _ = client.reset(skip_generation=False, track_string=waypoints)

            return isSuccess, temporary_images, data

        except Exception as e:
            # close the simulator
            self.simulator.tear_down()
            traceback.print_stack()
            # throw the exception
            raise e


    def perturb_tracks(
            self,
            track_name,
            daytime,
            weather,
            low_speed_threshold,
            low_speed_limit,
            total_crash_limit,
            max_gap,
            log_dir,
    ):

        env = UdacityGym(
            simulator=self.simulator,
        )
        if self.perturb:
            image_perturbation = self.setImagePerturbation()
        else:
            image_perturbation = None

        self.simulator.start()

        for perturbation in self.perturbation_functions:
            scale = self.start_scale
            crash_not_enough = crash_too_much = False

            while True:
                print("---------------------------------------")
                self.logger.info(f"Start perturbation: {perturbation}")
                self.logger.info(f"Current scale of testing: {scale}")

                obs, _ = env.reset(track=track_name, weather=weather, daytime=daytime)

                while not obs or not obs.is_ready():
                    obs = env.observe()
                    time.sleep(1)

                monitor = ImageCallBack(rows=self.image_size[0], cols=self.image_size[1]) if self.visualize else None

                if monitor:
                    monitor.display_waiting_screen()

                log_name = f"{track_name}_{perturbation}_scale{scale}_log.csv"
                log_path = os.path.join(log_dir, f"{track_name}/{track_name}_{perturbation}_scale{scale}_log")
                image_folder = os.path.join(log_path, "image_logs")

                data = []
                temporary_images = []
                low_speed_count, frame, manual_crash, total_crash = 0, 0, 0, 0
                skip_frames = 0
                prev_cte = None
                prev_is_crashed = None
                keep_running = True

                # First condition: keep testing in one round
                # Second: Ensure ADS doesn't stop in the middle
                # Third: Ensure ADS can still perform its driving skills
                while obs.lap == 1 and keep_running and total_crash <= total_crash_limit[1]:
                    frame += 1  # used to store images and csv_data
                    image_path = os.path.join(image_folder, f"{frame}.png")
                    # isinstance(obs.input_image, PIL.PngImagePlugin.PngImageFile)
                    # np.array(obs.input_image)为 RGB, shape (160, 320, 3)
                    if image_perturbation is not None:
                        image = image_perturbation.perturbation(np.array(obs.input_image), perturbation, scale)
                        obs.input_image = image  # isinstance(image, np.ndarray)
                    else:
                        image = obs.input_image

                    actions = self.agent(obs)  # obs.input_image.shape == (160, 320, 3)

                    if monitor:
                        monitor.display_img(obs.input_image, f"{actions.steering_angle}", f"{actions.throttle}",
                                            perturbation)

                    last_obs = obs
                    obs, reward, terminated, truncated, crash, info = env.step(actions)
                    # self.logger.info(obs.throttle,"   ,", obs.speed)
                    if skip_frames > 0:
                        skip_frames -= 1

                    # 道路内主观停止算作error type: 速度过低
                    if obs.speed <= low_speed_threshold and reward <= max_gap:
                        low_speed_count += 1
                    else:
                        low_speed_count = 0

                    # 连续20帧速度过低（小于0.001）则算作throttle prediction failure
                    if low_speed_count >= low_speed_limit:
                        self.logger.info("ADS speed too low! Quit and going to next scale level")
                        crash["low_speed"] += 1
                        keep_running = False

                    # Simulator未检测出的crash
                    if frame > 1 and abs(reward - prev_cte) > max_gap and crash["is_crashed"] != True and prev_is_crashed != True:
                        self.logger.info("Manual Crash detected!")
                        manual_crash += 1
                        crash["is_crashed"] = True
                    prev_cte = reward
                    prev_is_crashed = crash["is_crashed"]

                    if skip_frames == 0:
                        temporary_images.append((image_path, image))
                        data.append({
                            'index': frame,
                            'track': track_name,
                            'model': self.model_name,
                            'perturb_name': perturbation,
                            'scale': scale,
                            'image_path': image_path,
                            'lap': obs.lap,
                            'waypoint': obs.sector,
                            'speed': obs.speed,
                            'steer': actions.steering_angle,
                            'throttle': actions.throttle,
                            'cte': reward,
                            'manual_crash': manual_crash,
                            'out_of_track': crash.get("out_of_track"),
                            'collision': crash.get("collision"),
                            'low_speed': crash.get("low_speed"),
                            'is_crashed': crash.get("is_crashed"),
                        })

                    if crash.get("is_crashed"):
                        total_crash += 1
                        skip_frames = 10 # Skip the next 10 frames from recording

                    env.simulator.sim_state['is_crashed'] = False

                    while obs.time == last_obs.time:
                        obs = env.observe()
                        time.sleep(0.05)

                self.logger.info(f"Total crash at this crash is: {total_crash})")

                if monitor:
                    monitor.display_disconnect_screen()
                    monitor.destroy()

                if total_crash < total_crash_limit[0]:
                    crash_not_enough = True
                    if self.perturb:
                        self.logger.info(
                            f"ADS drives perfect! Skipping record of {log_name} and deleted the image folder")
                    else:
                        if not os.path.exists(image_folder):
                            os.makedirs(image_folder)
                            self.logger.info(f"Folder created: {image_folder}")
                        save_data_in_batch(log_name, log_path, data, temporary_images)
                        print("Normal driving saved in: ", {image_folder})
                        break

                elif total_crash_limit[0] <= total_crash <= total_crash_limit[1]:
                    if not os.path.exists(image_folder):
                        os.makedirs(image_folder)
                        self.logger.info(f"Folder created: {image_folder}")
                    save_data_in_batch(log_name, log_path, data, temporary_images)
                    self.logger.info(
                        f"Out_of_track Count: {crash.get('out_of_track')}; "
                        f"Collision Count: {crash.get('collision')}; "
                        f"Manual detected crash: {manual_crash}"
                    )
                    break  # jump out of current perturbation

                else:  # total_crash > total_crash_limit[1]:
                    if scale == 0:
                        self.logger.info("the lightest level of perturbation is already too much!")
                        break
                    crash_too_much = True
                    self.logger.info(
                        f"Too many crashes! Skipping record of {log_name} and deleted the image folder")

                if crash_not_enough and crash_too_much:
                    self.logger.info("Cannot find a solution between total_crash_limit")
                    break

                data.clear()
                temporary_images.clear()
                self.logger.info("Data has been cleared!")

                scale += 1
                if scale > self.max_scale:
                    break

        self.simulator.close()
        env.close()
        gc.collect()

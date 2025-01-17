import gc
from concurrent.futures import ThreadPoolExecutor

from perturbationdrive.Simulator.image_callback import ImageCallBack
from perturbationdrive.Simulator.Simulator import PerturbationSimulator
from perturbationdrive.operators.AutomatedDrivingSystem.ADS import ADS
from perturbationdrive.imageperturbations import ImagePerturbation
from perturbationdrive.Simulator.Scenario import Scenario
from perturbationdrive.RoadGenerator.RoadGenerator import RoadGenerator

from typing import List, Union, Dict, Tuple
import os, copy, time, csv
from PIL import Image
import numpy as np

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
        model: str = None,
        perturbation_functions: List[str] = None,
        attention_map: Dict = {},
        image_size: Tuple[int, int] = (160, 320),
        max_scale: int = 10,
        start_scale: int = 0,
        visualize: bool = True,
        perturb: bool = True,
    ):
        self.simulator = simulator
        self.agent = agent
        self.model = model
        self.perturbation_functions = perturbation_functions
        self.attention_map = attention_map
        self.image_size = image_size
        self.max_scale = max_scale
        self.start_scale = start_scale
        self.visualize = visualize
        self.perturb = perturb
    
    def setImagePerturbation(self):
        image_perturbation = ImagePerturbation(
            funcs=self.perturbation_functions,
            attention_map=self.attention_map,
            image_size=self.image_size,
        )
        return image_perturbation

    @staticmethod
    def save_image(image_path, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image.save(image_path)

    @staticmethod
    def perturb_driving_log(csv_path, data):
        with open(csv_path, 'w', newline='') as csvfile:
            if os.path.exists(csv_path):
                os.remove(csv_path)
                print(f"{csv_path} will be overwritten")

        with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if data:
                writer.writerow(data[0].keys())  # column names
                for row in data:
                    writer.writerow(row.values())

    def save_data_in_batch(self, log_name, log_path, data, image_data):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.save_image, image_path, image)
                for image_path, image in image_data
            ]
        # 等待所有任务完成
        for future in futures:
            try:
                future.result()  # 检查任务状态, 确保没有failed save
            except Exception as e:
                print(f"Error during image saving: {e}")

        self.perturb_driving_log(os.path.join(log_path, log_name), data)
        print(f"Data saved under {log_name}!")
        futures.clear()

    # def grid_seach(
    #         self,
    #         road_number: int = 0,
    #         road_generator: Union[RoadGenerator, None] = None,
    #         road_angles: List[int] = None,
    #         road_segments: List[int] = None,
    #         test_model: bool = False,
    #         weather: Union[str, None] = "Sun",
    #         weather_intensity: Union[int, None] = 90
    # ):
    #     """
    #     Basically, what we have done in image perturbations up until now but in a single nice function wrapped
    #
    #     If log_dir is none, we return the scenario outcomes
    #     """
    #     if self.perturb:
    #         image_perturbation = self.setImagePerturbation()
    #
    #     else:
    #         image_perturbation = None
    #
    #     scale = self.start_scale
    #     # perturbations: List[str] = []
    #     #
    #     # if perturb:
    #     #     perturbations: List[str] = copy.deepcopy(perturbation_functions)
    #
    #     # we append the empty perturbation here
    #     # perturbations.append("")
    #
    #     # set up simulator
    #     self.simulator.connect()
    #     # wait 1 second for connection to build up
    #     time.sleep(1)
    #
    #     # set up initial road
    #     waypoints = None
    #     if not road_generator is None:
    #         # TODO: Insert here all kwargs needed for specific generator
    #         waypoints = road_generator.generate(starting_pos=self.simulator.initial_pos, angles=road_angles,
    #                                             seg_lengths=road_segments)
    #
    #     # grid search loop
    #     while True:
    #         perturbation = self.perturbation_functions[0]
    #         print(
    #             f"{5 * '-'} Running Scenario: Perturbation {perturbation} on scale: {scale} {5 * '-'}"
    #         )
    #
    #         scenario = Scenario(
    #             waypoints=waypoints,
    #             perturbation_function=perturbation,
    #             perturbation_scale=scale,
    #         )
    #
    #         log_name = f"roadGen_{perturbation}_road{road_number}_scale{scale}_log.csv"
    #         log_path = f"./logs/RoadGenerator/{perturbation}/roadGen_{perturbation}_road{road_number}_scale{scale}_log"
    #         image_folder = os.path.join(self.log_path, "image_logs")
    #
    #         # simulate the scenario
    #         isSuccess, temporary_images, data = self.simulator.simulate_scanario(
    #             self.agent, scenario=scenario, perturbation_controller=image_perturbation, perturb=self.perturb, visualize=self.monitor,
    #             model_drive=test_model, weather=weather, intensity=weather_intensity, image_folder = image_folder
    #         )
    #
    #         # if len(perturbations) == 0:
    #         #     # all perturbations resulted in failures
    #         #     # we will still have one perturbation here because we never
    #         #     # drop the empty perturbation
    #         #     break
    #
    #         if isSuccess: # no crashed in this turn, so iterate into the next scale_level
    #             print(f"No crash in current scale: {scale}, increasing scale.")
    #             scale += 1
    #
    #         else:
    #             if len(temporary_images) > 50:# crash happens in current scale, record the image and data, jump out to the next perturbation
    #                 if not os.path.exists(image_folder):
    #                     os.makedirs(image_folder)
    #                 self.save_data_in_batch(self, self.log_name, self.log_path, data, temporary_images)
    #
    #             else:
    #                 print("Driving performance bad, too short! Data is not saving!")
    #
    #             scale = 0
    #             self.perturbation_functions.remove(perturbation)
    #             if len(self.perturbation_functions) == 0:
    #                 break
    #             data.clear()
    #             temporary_images.clear()
    #             time.sleep(2)
    #             print("Data has been cleared!")
    #
    #         if scale > self.max_scale:
    #             # we went through all scales
    #             print("Drives perfect in all scales! Going into the next perturbation!")
    #             break
    #
    #     # TODO: print command line summary of benchmarking process
    #     del image_perturbation
    #     del scenario
    #     del road_generator
    #
    #     # tear down the simulator
    #     self.simulator.tear_down()

    def perturb_tracks(
            self,
            track_name,
            daytime,
            weather,
            low_speed_threshold,
            low_speed_limit,
            total_crash_limit,
            log_dir,
    ):

        env = UdacityGym(
            simulator=self.simulator,
        )
        image_perturbation = self.setImagePerturbation()

        self.simulator.start()

        for perturbation in self.perturbation_functions:
            scale = self.start_scale
            crash_not_enough = crash_too_much = False

            while True:
                print("---------------------------------------")
                print("Start perturbation: ", perturbation)
                print("Current scale of testing: ", scale)

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
                    image = image_perturbation.perturbation(np.array(obs.input_image), perturbation, scale)
                    obs.input_image = image  # isinstance(image, np.ndarray)

                    actions = self.agent(obs)  # obs.input_image.shape == (160, 320, 3)

                    if monitor:
                        monitor.display_img(obs.input_image, f"{actions.steering_angle}", f"{actions.throttle}",
                                            perturbation)

                    last_obs = obs
                    obs, reward, terminated, truncated, crash, info = env.step(actions)
                    # print(obs.throttle,"   ,", obs.speed)
                    if skip_frames > 0:
                        skip_frames -= 1

                    # 道路内主观停止算作error type: 速度过低
                    if obs.speed <= low_speed_threshold and reward <= 4:
                        low_speed_count += 1
                    else:
                        low_speed_count = 0

                    # 连续20帧速度过低（小于0.001）则算作throttle prediction failure
                    if low_speed_count >= low_speed_limit:
                        print("ADS speed too low! Quit and going to next scale level")
                        crash["low_speed"] += 1
                        keep_running = False

                    # Simulator未检测出的crash
                    if frame > 1 and abs(reward - prev_cte) > 4 and crash["is_crashed"] != True and prev_is_crashed != True:
                        print("Manual Crash detected!")
                        manual_crash += 1
                        crash["is_crashed"] = True
                    prev_cte = reward
                    prev_is_crashed = crash["is_crashed"]

                    if skip_frames == 0:
                        temporary_images.append((image_path, image))
                        data.append({
                            'index': frame,
                            'track': track_name,
                            'model': self.model,
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

                print("Total crash at this crash is: ", {total_crash})

                if monitor:
                    monitor.display_disconnect_screen()
                    monitor.destroy()

                if total_crash < total_crash_limit[0]:
                    crash_not_enough = True
                    print(
                        f"ADS drives perfect! Skipping record of {log_name} and deleted the image folder")

                elif total_crash_limit[0] <= total_crash <= total_crash_limit[1]:
                    if not os.path.exists(image_folder):
                        os.makedirs(image_folder)
                        print("Folder created: ", image_folder)
                    self.save_data_in_batch(log_name, log_path, data, temporary_images)
                    print("Out_of_track Count: ", crash.get("out_of_track"), "; Collision Count: ",
                          crash.get("collision"), "Manual detected crash: ", manual_crash)

                    break  # jump out of current perturbation

                else:  # total_crash > total_crash_limit[1]:
                    if scale == 0:
                        print("the lightest level of perturbation is already too much!")
                        break
                    crash_too_much = True
                    print(
                        f"Too many crashes! Skipping record of {log_name} and deleted the image folder")

                if crash_not_enough and crash_too_much:
                    print("Cannot find a solution between total_crash_limit")
                    break

                data.clear()
                temporary_images.clear()
                print("Data has been cleared!")

                scale += 1
                if scale > self.max_scale:
                    break

        self.simulator.close()
        env.close()
        gc.collect()

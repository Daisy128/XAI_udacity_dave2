import gc
import os
import csv
import time
import logging
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from utils.conf import track_infos, perturb_cfgs
from udacity_gym.agent_tf import SupervisedAgent_tf
from udacity_gym import UdacitySimulator, UdacityGym
from perturbationdrive.imageperturbations import ImagePerturbation
from perturbationdrive.Simulator.image_callback import ImageCallBack

def save_image(image_path, image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(image_path)

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

def save_data_in_batch(log_name, log_path, data, image_data):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(save_image, image_path, image)
            for image_path, image in image_data
        ]
    # 等待所有任务完成
    for future in futures:
        try:
            future.result()  # 检查任务状态, 确保没有failed save
        except Exception as e:
            print(f"Error during image saving: {e}")

    perturb_driving_log(os.path.join(log_path, log_name), data)
    print(f"Data saved under {log_name}!")
    futures.clear()

def perturbed_simulate(config):

    TRACK = track_infos[config['track_index']]['track_name']
    DAYTIME = "day"
    WEATHER = "sunny"
    SCALE = perturb_cfgs['max_scale']
    IMAGE_SIZE = (perturb_cfgs['image_height'], perturb_cfgs['image_width'])
    LOW_SPEED_THRESHOLD = perturb_cfgs['low_speed_threshold']
    LOW_SPEED_LIMIT = perturb_cfgs['low_speed_limit']
    TOTAL_CRASH_LIMIT = perturb_cfgs['total_crash_limit'][TRACK]

    # Creating the simulator wrapper
    simulator = UdacitySimulator(
        sim_exe_path=track_infos[config['track_index']]['simulator']['exe_path'],
        host=track_infos[config['track_index']]['simulator']['host'],
        port=track_infos[config['track_index']]['simulator']['port'],
    )
    # Creating the gym environment
    env = UdacityGym(
        simulator=simulator,
    )
    # log_observation_callback = LogObservationCallback(log_directory)
    agent = SupervisedAgent_tf(
        model_path=config['model_path'],
        max_speed=40,
        min_speed=6,
        predict_throttle=True
    )
    image_perturbation = ImagePerturbation(
        funcs=config['perturbations'],
        attention_map={},
        image_size=IMAGE_SIZE
    )

    simulator.start()

    for perturbation in config['perturbations']:
        scale = config['start_scale']

        crash_not_enough = crash_too_much = False
        while True:
            print("---------------------------------------")
            print("Start perturbation: ", perturbation)
            print("Current scale of testing: ", scale)

            obs, _ = env.reset(track=TRACK, weather=WEATHER, daytime=DAYTIME)

            while not obs or not obs.is_ready():
                obs = env.observe()
                time.sleep(1)

            monitor = ImageCallBack(rows=IMAGE_SIZE[0], cols=IMAGE_SIZE[1]) if perturb_cfgs['visualize'] else None
            if monitor:
                monitor.display_waiting_screen()

            # LOG_PATH here should be ABSOLUTE, info is required in csv log file
            LOG_NAME = f"{TRACK}_{perturbation}_scale{scale}_log.csv"
            LOG_PATH = f"perturbationdrive/logs/{TRACK}/{TRACK}_{perturbation}_scale{scale}_log"

            image_folder = os.path.join(LOG_PATH, "image_logs")

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
            while obs.lap == 1 and keep_running and total_crash <= TOTAL_CRASH_LIMIT[1]:
                frame += 1  # used to store images and csv_data
                image_path = os.path.join(image_folder, f"{frame}.png")
                # isinstance(obs.input_image, PIL.PngImagePlugin.PngImageFile)
                # np.array(obs.input_image)为 RGB, shape (160, 320, 3)
                image = image_perturbation.perturbation(np.array(obs.input_image), perturbation, scale)
                obs.input_image = image  # isinstance(image, np.ndarray)

                actions = agent(obs)  # obs.input_image.shape == (160, 320, 3)

                if monitor:
                    monitor.display_img(obs.input_image, f"{actions.steering_angle}", f"{actions.throttle}",
                                        perturbation)

                last_obs = obs
                obs, reward, terminated, truncated, crash, info = env.step(actions)
                # print(obs.throttle,"   ,", obs.speed)
                if skip_frames > 0:
                    skip_frames -= 1

                # 道路内主观停止算作error type: 速度过低
                if obs.speed <= LOW_SPEED_THRESHOLD and reward <= 2:
                    low_speed_count += 1
                else:
                    low_speed_count = 0

                # 连续20帧速度过低（小于0.001）则算作throttle prediction failure
                if low_speed_count >= LOW_SPEED_LIMIT:
                    crash["low_speed"] += 1
                    keep_running = False

                # Simulator未检测出的crash
                if frame > 1 and abs(reward - prev_cte) > 2 and crash["is_crashed"] != True and prev_is_crashed != True:
                    print("Manual Crash detected!")
                    manual_crash += 1
                    crash["is_crashed"] = True
                prev_cte = reward
                prev_is_crashed = crash["is_crashed"]

                if skip_frames == 0:
                    temporary_images.append((image_path, image))
                    data.append({
                        'index': frame,
                        'track': TRACK,
                        'model': config['model_name'],
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
                    skip_frames = 5

                env.simulator.sim_state['is_crashed'] = False

                while obs.time == last_obs.time:
                    obs = env.observe()
                    time.sleep(0.05)

            print("Total crash at this crash is: ", {total_crash})

            if monitor:
                monitor.display_disconnect_screen()
                monitor.destroy()

            if total_crash < TOTAL_CRASH_LIMIT[0]:
                crash_not_enough = True
                print(
                    f"ADS drives perfect! Skipping record of {LOG_NAME} and deleted the image folder")

            elif TOTAL_CRASH_LIMIT[0] <= total_crash <= TOTAL_CRASH_LIMIT[1]:
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                    print("Folder created: ", image_folder)
                save_data_in_batch(LOG_NAME, LOG_PATH, data, temporary_images)
                print("Out_of_track Count: ", crash.get("out_of_track"), "; Collision Count: ",
                      crash.get("collision"), "Manual detected crash: ", manual_crash)

                break  # jump out of current perturbation

            else:  # total_crash > TOTAL_CRASH_LIMIT[1]:
                if scale == 0:
                    print("the lightest level of perturbation is already too much!")
                    break
                crash_too_much = True
                print(
                    f"Too many crashes! Skipping record of {LOG_NAME} and deleted the image folder")

            if crash_not_enough and crash_too_much:
                print("Cannot find a solution between TOTAL_CRASH_LIMIT")
                break

            data.clear()
            temporary_images.clear()
            print("Data has been cleared!")

            scale += 1
            if scale > SCALE:
                break

    simulator.close()
    env.close()
    gc.collect()
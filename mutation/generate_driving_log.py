import gc
import json
import os
import pathlib
import time

import numpy as np
import tqdm
from udacity_gym import UdacitySimulator, UdacityGym
from udacity_gym.agent_tf import SupervisedAgent_tf
from utils.conf import track_infos
from perturbationdrive.perturbationdrive import save_data_in_batch

track_index = 1

def generate_driving_log(env, mutate_path, log_path):

    observation, _ = env.reset(track=f"{track}", weather=f"{weather}", daytime=f"{daytime}")

    # Wait for environment to set up
    while not observation or not observation.is_ready():
        observation = env.observe()
        print("Waiting for environment to set up...")
        time.sleep(1)

    agent = SupervisedAgent_tf(
        model_path=mutate_path,
        max_speed=40,
        min_speed=6,
        predict_throttle=True
    )

    frame, skip_frames = 0, 0
    temporary_images = []
    data = []
    manual_crash = 0
    max_gap = 2
    prev_cte = None
    prev_is_crashed = None
    total_crash = 0

    log_name = f"{track}_{mutation_name}_log.csv"
    image_folder = os.path.join(log_path, "image_logs")

    # Interacting with the gym environment
    while observation.lap == 1:
        frame += 1
        image_path = os.path.join(image_folder, f"{frame}.png")

        action = agent(observation)
        last_observation = observation
        observation, reward, terminated, truncated, crash, info = env.step(action)
        image = observation.input_image

        if skip_frames > 0:
            skip_frames -= 1

        if frame > 1 and abs(reward - prev_cte) > max_gap and crash["is_crashed"] != True and prev_is_crashed != True:
            print("Manual Crash detected!")
            manual_crash += 1
            crash["is_crashed"] = True
        prev_cte = reward
        prev_is_crashed = crash["is_crashed"]

        if skip_frames == 0:
            temporary_images.append((image_path, image))
            data.append({
                'frameId': frame,
                'track': track,
                'model': mutate_path,
                'perturb_name': mutation_name,
                'scale': "no scale",
                'image_path': image_path,
                'lap': observation.lap,
                'waypoint': observation.sector,
                'speed': observation.speed,
                'steer': action.steering_angle,
                'throttle': action.throttle,
                'cte': reward,
                'manual_crash': manual_crash,
                'out_of_track': crash.get("out_of_track"),
                'collision': crash.get("collision"),
                'low_speed': crash.get("low_speed"),
                'is_crashed': crash.get("is_crashed"),
            })

        if crash.get("is_crashed"):
            total_crash += 1
            skip_frames = 10  # Skip the next 10 frames from recording

        env.simulator.sim_state['is_crashed'] = False

        while observation.time == last_observation.time:
            observation = env.observe()
            time.sleep(0.005)

    os.makedirs(image_folder, exist_ok=True)
    save_data_in_batch(log_name, log_path, data, temporary_images)

    data.clear()
    temporary_images.clear()
    gc.collect()

if __name__ == '__main__':

    # Track settings
    track = track_infos[track_index]['track_name'] # lake, mountain
    daytime = "day"
    weather = "sunny"

    simulator = UdacitySimulator(
        sim_exe_path=track_infos[track_index]['simulator']['exe_path'],
        host=track_infos[track_index]['simulator']['host'],
        port=track_infos[track_index]['simulator']['port'],
    )

    env = UdacityGym(
        simulator=simulator,
    )

    mutate_root_dir = pathlib.Path(f"model/ckpts/ads-mutation")
    log_root_dir = pathlib.Path(f"mutation/logs/{track}")

    simulator.start()

    for mutation_name in os.listdir(mutate_root_dir):
        # model/ckpts/ads-mutation/add_weights_regularisation_l1_6
        mutate_models_path = os.path.join(mutate_root_dir, mutation_name, f"track{track_index}_{track}")

        models = [f for f in os.listdir(mutate_models_path) if os.path.isfile(os.path.join(mutate_models_path, f))]
        # xxx_final.h5
        model_name = models[0]

        mutate_path = os.path.join(mutate_models_path, model_name)
        log_path = os.path.join(log_root_dir, f"{track}_{mutation_name}_log")

        if os.path.exists(log_path):
            print(f"{log_path} already exists, skip!")
            continue
        else:
            os.makedirs(log_path, exist_ok=True)
            print(f"{log_path} created! Recording driving log on: ", mutate_path)

            generate_driving_log(env, mutate_path, log_path)

    simulator.close()
    env.close()
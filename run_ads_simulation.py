import json
import pathlib
import time
import tqdm
from udacity_gym import UdacitySimulator, UdacityGym
from udacity_gym.agent_tf import SupervisedAgent_tf
from utils.conf import track_infos

track_index = 1

if __name__ == '__main__':

    # Track settings
    track = track_infos[track_index]['track_name'] # lake, mountain
    daytime = "day"
    weather = "sunny"
    log_directory = pathlib.Path(f"udacity_dataset_lake_dave/{track}_{weather}_{daytime}")
    print(track_infos[track_index]['simulator'])

    # Creating the simulator wrapper
    simulator = UdacitySimulator(
        sim_exe_path=track_infos[track_index]['simulator']['exe_path'],
        host=track_infos[track_index]['simulator']['host'],
        port=track_infos[track_index]['simulator']['port'],
    )
    # Creating the gym environment
    env = UdacityGym(
        simulator=simulator,
    )
    observation, _ = env.reset(track=f"{track}", weather=f"{weather}", daytime=f"{daytime}")

    simulator.start()

    # Wait for environment to set up
    while not observation or not observation.is_ready():
        observation = env.observe()
        print("Waiting for environment to set up...")
        time.sleep(1)

    # log_observation_callback = LogObservationCallback(log_directory)
    agent = SupervisedAgent_tf(
        model_path=track_infos[track_index]['model_path'],
        max_speed=40,
        min_speed=6,
        predict_throttle=True
    )

    # Interacting with the gym environment
    for _ in tqdm.tqdm(range(200)):
        action = agent(observation)
        last_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)

        while observation.time == last_observation.time:
            observation = env.observe()
            time.sleep(0.005)

    if info:
        json.dump(info, open(log_directory.joinpath("info.json"), "w"))

    # log_observation_callback.save()
    simulator.close()
    env.close()
    print("Experiment concluded.")

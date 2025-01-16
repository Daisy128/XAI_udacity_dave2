import gc
import time
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from perturbationdrive import ImageCallBack
from perturbationdrive.utils.image_log import *
from utils.conf import track_infos, perturb_cfgs
from udacity_gym.agent_tf import SupervisedAgent_tf
from udacity_gym import UdacitySimulator, UdacityGym
from perturbationdrive.imageperturbations import ImagePerturbation

def perturbed_simulate(data):

    track_index = data['track_index']
    model_name = data['model_name']
    model_path = data['model_path']
    perturbations = data['perturbations']
    
    TRACK = track_infos[track_index]['track_name']  # lake, mountain
    DAYTIME = "day"
    WEATHER = "sunny"
    SCALE = perturb_cfgs['scale']
    IMAGE_SIZE = (perturb_cfgs['image_height'], perturb_cfgs['image_width'])
    LOW_SPEED_THRESHOLD = perturb_cfgs['low_speed_threshold']
    LOW_SPEED_LIMIT = perturb_cfgs['low_speed_limit']
    TOTAL_CRASH_LIMIT = perturb_cfgs['total_crash_limit'][TRACK]

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
    # log_observation_callback = LogObservationCallback(log_directory)
    agent = SupervisedAgent_tf(
        model_path=model_path,
        max_speed=40,
        min_speed=6,
        predict_throttle=True
    )
    image_perturbation = ImagePerturbation(
        funcs=perturbations,
        attention_map={},
        image_size=IMAGE_SIZE
    )

    simulator.start()
    
    for perturbation in perturbations:
        scale = data['start_index']
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
            LOG_PATH = f"/home/jiaqq/Documents/ThirdEye-II/perturbationdrive/logs/{TRACK}/{TRACK}_{perturbation}_scale{scale}_log"

            image_folder = os.path.join(LOG_PATH, "image_logs")

            data = []
            temporary_images = []
            low_speed_count, frame, total_crash = 0, 0, 0
            keep_running = True

            # First condition: keep testing in one round
            # Second: Ensure ADS doesn't stop in the middle
            # Third: Ensure ADS can still perform its driving skills
            while obs.lap == 1 and keep_running and total_crash <= TOTAL_CRASH_LIMIT[1]:
                frame += 1 # used to store images and csv_data
                image_path = os.path.join(image_folder, f"{frame}.png")

                # isinstance(obs.input_image, PIL.PngImagePlugin.PngImageFile)
                # np.array(obs.input_image)为 RGB, shape (160, 320, 3)
                image = image_perturbation.perturbation(np.array(obs.input_image), perturbation, scale)
                obs.input_image = image # isinstance(image, np.ndarray)

                temporary_images.append((image_path, image))

                actions = agent(obs) # obs.input_image.shape == (160, 320, 3)

                if monitor:
                    monitor.display_img(obs.input_image, f"{actions.steering_angle}", f"{actions.throttle}", perturbation)

                last_obs = obs
                obs, reward, terminated, truncated, crash, info  = env.step(actions)
                # print(obs.throttle,"   ,", obs.speed)

                if obs.speed <= LOW_SPEED_THRESHOLD and reward <= 4: # 只算道路内主观停止
                    low_speed_count += 1
                else:
                    low_speed_count = 0

                #     def ————（speed, threshold, cte）3种情况 停止 倒车 卡住（cte相关）

                if low_speed_count >= LOW_SPEED_LIMIT: # 连续20帧速度小于0.001
                    crash["low_speed"] += 1
                    keep_running = False

                data.append({
                    'index': frame,
                    'track': TRACK,
                    'model': model_name,
                    'perturb_name': perturbation,
                    'scale': scale,
                    'image_path': image_path,
                    'lap': obs.lap,
                    'waypoint': obs.sector,
                    'speed': obs.speed,
                    'steer': actions.steering_angle,
                    'throttle': actions.throttle,
                    'cte': reward,
                    'out_of_track': crash.get("out_of_track"),
                    'collision': crash.get("collision"),
                    'low_speed': crash.get("low_speed"),
                    'is_crashed': crash.get("is_crashed")
                })

                if crash.get("is_crashed"):
                    total_crash += 1

                env.simulator.sim_state['is_crashed'] = False

                while obs.time == last_obs.time:
                    obs = env.observe()
                    time.sleep(0.05)

            print("Total crash at this crash is: ", {total_crash})

            if monitor:
                monitor.display_disconnect_screen()
                monitor.destroy()

            if TOTAL_CRASH_LIMIT[0] <= total_crash <= TOTAL_CRASH_LIMIT[1]:
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(save_image, image_path, image) for image_path, image in temporary_images]
                # 等待所有任务完成
                for future in futures:
                    try:
                        future.result()  # 检查任务状态
                    except Exception as e:
                        print(f"Error during image saving: {e}")

                perturb_driving_log(os.path.join(LOG_PATH, LOG_NAME), data)

                print(f"Data saved under {TRACK}_{perturbation}_scale{scale}_log.csv!")
                print("Out_of_track Count: ", crash.get("out_of_track"), "; Collision Count: ",
                      crash.get("collision"))

                data.clear()
                futures.clear()
                temporary_images.clear()
                print("Data has been cleared!")

                break  # jump out of current perturbation


            elif total_crash < TOTAL_CRASH_LIMIT[0]:
                crash_not_enough = True
                print(f"ADS drives perfect! Skipping record of {TRACK}_{perturbation}_scale{scale}_log.csv and deleted the image folder")

            else:
                if scale == 0:
                    print("the lightest level of perturbation is already too much!")
                    break
                crash_too_much = True
                print(f"Too many crashes! Skipping record of {TRACK}_{perturbation}_scale{scale}_log.csv and deleted the image folder")

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
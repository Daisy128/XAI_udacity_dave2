import base64
import time
from io import BytesIO
from multiprocessing import Process
from threading import Thread

import PIL
import eventlet
eventlet.monkey_patch()
import numpy as np
from PIL import Image
from flask import Flask
from flask_socketio import SocketIO

from .action import UdacityAction
from .logger import CustomLogger
from .observation import UdacityObservation
class UdacityExecutor:
    # TODO: avoid cycles

    def __init__(
            self,
            host: str = "127.0.0.1",
            port: int = 4567,
    ):
        # Simulator network settings
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.sio = SocketIO(
            self.app,
            async_mode='eventlet',
            cors_allowed_origins="*",
            transports=['websocket'],
        )
        # Socket IO callbacks
        self.sio.on('connect')(self.on_connect)
        self.sio.on('car_telemetry')(self.on_telemetry)
        self.sio.on('episode_metrics')(self.on_episode_metrics)
        self.sio.on('episode_events')(self.on_episode_events)
        self.sio.on('episode_event')(self.on_episode_event)
        self.sio.on('sim_paused')(self.on_sim_paused)
        self.sio.on('sim_resumed')(self.on_sim_resumed)

        # Simulator logging
        self.logger = CustomLogger(str(self.__class__))
        # Simulator
        from .simulator import simulator_state
        self.sim_state = simulator_state
        # Manage connection in separate process
        self.client_thread = Process(target=self._start_server)
        self.client_thread.daemon = True
        self.sim_state['done'] = False

    # 接收车辆数据，所以每个observation后都要处理并生成更新sim_state
    def on_telemetry(self, data):
        # self.logger.info(f"Received data from udacity client: {data}")
        #print("on_telemetry")
        # TODO: check data image, verify from sender that is not empty
        try:
            input_image = Image.open(BytesIO(base64.b64decode(data["image"])))
        except PIL.UnidentifiedImageError:
            print("Front facing camera image UnidentifiedImageError.")
            input_image = None

        try:
            semantic_segmentation = Image.open(BytesIO(base64.b64decode(data["semantic_segmentation"])))
        except PIL.UnidentifiedImageError:
            #print("Segmentation camera image UnidentifiedImageError.")
            semantic_segmentation = None

        #print(data)
        observation = UdacityObservation(
            input_image=input_image,
            semantic_segmentation=semantic_segmentation,
            position=(float(data["pos_x"]), float(data["pos_y"]), float(data["pos_z"])),
            steering_angle=float(self.sim_state.get('action', None).steering_angle),
            throttle=float(self.sim_state.get('action', None).throttle),
            lap=int(data['lap']),
            sector=int(data['sector']),
            speed=float(data["speed"]) * 3.6,  # conversion m/s to km/h
            cte=float(data["cte"]),
            next_cte=float(data["next_cte"]),
            time=int(time.time() * 1000)
        )
        self.sim_state['observation'] = observation

        # Sending control
        self.send_control()

        if self.sim_state.get('paused', False):
            self.send_pause()
        else:
            self.send_resume()

        # if self.sim_state['done']:
        #     print("Reach the limits")
        #     self.send_reset()

        track_info = self.sim_state.get('track', None)
        if track_info:
            track, weather, daytime = track_info['track'], track_info['weather'], track_info['daytime']
            self.send_track(track, weather, daytime)
            self.sim_state['track'] = None

    def on_connect(self):
        self.logger.info("Udacity client connected")
        track_info = self.sim_state.get('track', None)
        # TODO: do it in a better way
        while not track_info:
            time.sleep(1)
            track_info = self.sim_state.get('track', None)
        track, weather, daytime = track_info['track'], track_info['weather'], track_info['daytime']
        self.send_track(track, weather, daytime)
        self.sim_state['track'] = None
        self.sim_state['out_of_track'] = self.sim_state['collision'] = self.sim_state['low_speed'] = 0
        self.sim_state['is_crashed'] = False

    def on_sim_paused(self, data):
        self.sim_state['sim_state'] = 'paused'

    def on_sim_resumed(self, data):
        # TODO: change 'running' with ENUM
        self.sim_state['sim_state'] = 'running'

    # 一个simulator process开始和结束时返回total collision和out_of_track
    def on_episode_metrics(self, data):
        self.logger.info(f"episode metrics {data}")
        print("Episode metrics:" , data)
        self.sim_state['episode_metrics'] = data

    # 一个simulator process开始和结束时返回空字典
    def on_episode_events(self, data):
        self.logger.info(f"episode events {data}")
        self.sim_state['events'] += [data]
        print("Episode events:", data)

    def on_episode_event(self, data):
        # 模拟器在此处返回新的log, 说明有episode event发生
        self.logger.info(f"episode event {data}")

        self.sim_state['crash_log'] = data.get('key')

        if str(self.sim_state['crash_log']) == "out_of_track":
            self.sim_state['out_of_track'] += 1
        elif str(self.sim_state['crash_log']) == "collision":
            self.sim_state['collision'] += 1
        elif self.sim_state['crash_log'] is not None:
            self.sim_state['other_crash'] += 1

        if self.sim_state['crash_log'] is not None:
            self.sim_state['is_crashed'] = True

        self.sim_state['events'] += [data]
        # print(data) # {'timestamp': '1732723313733', 'key': 'out_of_track', 'value': ''}

    def send_control(self) -> None:
        # self.logger.info(f"Sending control")
        action: UdacityAction = self.sim_state.get('action', None)
        if action:
            self.sio.emit(
                "action",
                data={
                    "steering_angle": action.steering_angle.__str__(),
                    "throttle": action.throttle.__str__(),
                },
                skip_sid=True,
            )
            eventlet.sleep(0)

    def send_pause(self):
        self.sio.emit("pause_sim", skip_sid=True)

    def send_resume(self):
        # print("send resume")
        self.sio.emit("resume_sim", skip_sid=True)
        if self.sim_state.get('crash_log') is not None:
            self.sim_state['crash_log'] = None

    def send_track(self, track, weather, daytime):
        self.sio.emit("end_episode", skip_sid=True)
        self.sio.emit("start_episode", data={
            "track_name": track,
            "weather_name": weather,
            "daytime_name": daytime,
        }, skip_sid=True)

    def send_reset(self) -> None:
        self.sio.emit("reset", data={}, skip_sid=True)
        # print("Reset", end="\n", flush=True)

    def start(self):
        # Start Socket IO Server in separate thread
        self.client_thread.start()

    def _start_server(self):
        self.sio.run(self.app, host=self.host, port=self.port)

    def close(self):
        self.sio.stop()


if __name__ == '__main__':
    sim_executor = UdacityExecutor()
    sim_executor.start()

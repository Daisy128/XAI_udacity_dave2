# used modules from perturbation drive
from PIL import Image
from numpy import ndarray, uint8
import os

#import cvxpy as cp
from perturbationdrive.Simulator.Simulator import PerturbationSimulator
from perturbationdrive import GlobalLog as Gl
import traceback

# used libraries
from perturbationdrive.RoadGenerator.udacity_utils.envs.udacity.udacity_gym_env import (
    UdacityGymEnv_RoadGen,
)
from typing import Union, Tuple
import cv2
import gym
import numpy as np
import time
import math

WAYPOINT_THRESHOLD = 5
ANGLE_THRESHOLD = 0


class Waypoint_control_utils():
        def __init__(self,treshold,angle_treshold):
            self.waypoint_treshold=treshold
            self.angle_treshold=angle_treshold

        def angle_difference(self,a1, a2):
            diff = a1 - a2
            if diff>=180:
                diff-=360

            
            return diff
        
        def convert_waypoints(self,input_string):
            # Split the input string by the '@' symbol to get individual waypoints
            waypoints = input_string.split('@')
            
            # Initialize an empty list to hold the converted waypoints
            waypoint_list = []
            
            # Iterate through each waypoint string
            for waypoint in waypoints:
                # Split the string by the ',' symbol to get x, y, z values
                x, z, y = waypoint.split(',')
                
                # Convert x, y, z to floats and rearrange to [x, y, z]
                waypoint_list.append([float(x), float(y)])

            # waypoint_list,_=self.generate_road_margins(waypoint_list,1)
            
            return waypoint_list[1:]
        
        def generate_road_margins(self,road_points, offset):
            left_margins = []
            right_margins = []

            num_points = len(road_points)

            # Calculate the direction vectors for each road segment
            direction_vectors = []
            for i in range(num_points - 1):
                dx = road_points[i + 1][0] - road_points[i][0]
                dy = road_points[i + 1][1] - road_points[i][1]
                mag = np.sqrt(dx ** 2 + dy ** 2)
                direction_vectors.append((dx / mag, dy / mag))

            # Average neighboring direction vectors to get smoother normals
            averaged_directions = []
            for i in range(num_points - 1):
                if i == 0:
                    averaged_directions.append(direction_vectors[0])
                elif i == num_points - 2:
                    averaged_directions.append(direction_vectors[-1])
                else:
                    averaged_directions.append(((direction_vectors[i][0] + direction_vectors[i - 1][0]) / 2,
                                                (direction_vectors[i][1] + direction_vectors[i - 1][1]) / 2))

            # Calculate normals and generate margins
            for i in range(num_points - 1):
                dx, dy = averaged_directions[i]
                nx = -dy
                ny = dx

                left_x = road_points[i][0] + offset * nx
                left_y = road_points[i][1] + offset * ny
                right_x = road_points[i][0] - offset * nx
                right_y = road_points[i][1] - offset * ny

                left_margins.append([left_x, left_y])
                right_margins.append([right_x, right_y])

            return left_margins, right_margins
    
        def angle_extraction(self, x1, y1, z1, x2, y2, z2):

            # Calculate the distances between points
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1

            # Calculate the angles on each axis
            angle_x_axis = math.atan2(dy, dz)
            angle_y_axis = math.atan2(dx, dz)
            angle_z_axis = math.atan2(dy, dx)

            # Convert angles from radians to degrees
            angle_x_axis_degrees = math.degrees(angle_x_axis)
            angle_y_axis_degrees = math.degrees(angle_y_axis)
            angle_z_axis_degrees = math.degrees(angle_z_axis)
            return angle_x_axis_degrees, angle_y_axis_degrees, angle_z_axis_degrees
        
        def exponential_increase(self,number, factor):
            return factor * (1 - np.exp(-number))

        def calculate_control(self, x_target, y_target, simulator_pose, simulator_orientation):
            x_cur, y_cur, _ = simulator_pose

            #print(f"Position\nx:{x_cur}, y:{y_cur}")
            _, angle_cur, _ = simulator_orientation
            #print(f"Orientation: {round(angle_cur, 3)}, {round(math.radians(angle_cur), 3)} rad")
            distance = math.sqrt((x_target - x_cur)**2 + (y_target - y_cur)**2)
            #print(f"dist {distance}")
            _, angle_y_axis_degrees, _=self.angle_extraction(x_cur, 0.0, y_cur, x_target, 0.0, y_target)
            #print(f"Angle to goal: {angle_y_axis_degrees}")
            angle_difference=self.angle_difference(angle_cur,angle_y_axis_degrees)
            # print(f"angle diff {angle_difference}")
            steering = (math.radians(-angle_difference) / math.pi)*4
            # print(f"angle diff: {angle_difference}, steering: {steering}")
            throttle=distance/10

            return steering, throttle, distance, angle_difference,angle_y_axis_degrees
        
        def calculate_distance(self, x_target, y_target, simulator_pose):
            x_cur, y_cur, _ = simulator_pose
            distance = math.sqrt((x_target - x_cur)**2 + (y_target - y_cur)**2)
            return distance

        def is_waypoint_in_back(self,current_pos, current_orientation, waypoint_x,waypoint_y):
            current_x, current_y, _ = current_pos
            _, angle_cur, _ = current_orientation
            current_orientation=math.radians(angle_cur)
            
            # Calculate vector to the next waypoint
            waypoint_vector = (waypoint_x - current_x, waypoint_y - current_y)
            
            # Calculate vehicle forward vector based on its orientation
            vehicle_forward_vector = (math.cos(current_orientation), math.sin(current_orientation))
            
            # Calculate the dot product
            dot_product = waypoint_vector[0] * vehicle_forward_vector[0] + waypoint_vector[1] * vehicle_forward_vector[1]
            
            # If the dot product is positive, the waypoint is in front; otherwise, it's behind
            return dot_product <= 0


class UdacitySimulator(PerturbationSimulator):
    def __init__(
        self,
        simulator_exe_path: str = "./examples/udacity/udacity_utils/sim/udacity_sim.app",
        host: str = "127.0.0.1",
        port: int = 9091,
        show_image_cb=True
    ):
        # udacity road is 8 units wide
        super().__init__(
            max_xte=4.0,
            simulator_exe_path=simulator_exe_path,
            host=host,
            port=port,
            initial_pos=None,
        )
        self.client: Union[UdacityGymEnv_RoadGen, None] = None
        self.logger = Gl("UdacitySimulator")
        self.show_image_cb=show_image_cb
            

    def connect(self):
        super().connect()
        self.client = UdacityGymEnv_RoadGen(
            seed=1,
            exe_path=self.simulator_exe_path,
        )
        self.client.reset()
        time.sleep(2)
        self.logger.info("Connected to Udacity Simulator")
        # set initial pos
        obs, done, info = self.client.observe()
        x, y, z = info["pos"]
        if self.initial_pos is None:
            self.initial_pos = (x, y, z, 2 * self.max_xte)
        self.logger.info(f"Initial pos: {self.initial_pos}")

    def tear_down(self):
        self.client.close()

    def name(self) -> str:
        return "UdacitySimualtorAdapter"

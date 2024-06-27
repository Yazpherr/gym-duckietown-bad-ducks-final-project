#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import argparse
import pyglet
# from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
import numpy as np
import random
import time
global error_rot
global error_vel
global error_line
global x_objetivo
global y_objetivo
global x_inicial
global y_inicial
global Kd
global Ki
global Kp

#Variables del controlador
Kp=0.8
Ki=0
Kd=0

x_objetivo=1.7
y_objetivo=0.4
x_inicial=0.1
y_inicial=0.3

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_vel = 0
        self.previous_error = 0
        self.integral_rot = 0
        self.previous_error_rot = 0
        self.previous_error_vel = 0

    def update_err(self, error_vel, error_rot):
        self.integral_vel += error_vel
        self.integral_rot += error_rot
        derivative_vel = error_vel - self.previous_error_vel
        derivative_rot = error_rot - self.previous_error_rot
        control_rot = self.Kp * error_rot + self.Ki * self.integral_rot + self.Kd * derivative_rot
        control_vel = self.Kp * error_vel + self.Ki * self.integral_vel + self.Kd * derivative_vel
        self.previous_error_rot = error_rot
        self.previous_error_vel = error_vel
        return control_vel, control_rot
    def reset_error(self):
        self.previous_error=0
        self.integral=0


class LineFollower:
    def __init__(self):
        # Initialize variables
        self.last_error = 0
        self.integral = 0

    def follow_line(self, line_position):
        # Error calculation
        error_line = line_position - env.cur_pos[2]

        # Proportional term
        proportional = Kp * error_line

        # Integral term
        self.integral += error_line
        integral = Ki * self.integral

        # Derivative term
        derivative = Kd * (error_line - self.last_error)
        self.last_error = error_line

        # PID control output
        action_control = proportional + integral + derivative

        return action_control

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='linea_recta_PID')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
    )
    PIDControl= PIDController(Kp,Ki,Kd)
    PID_Line_Follower = LineFollower()
    env.cur_pos=[x_inicial,0,y_inicial]
    env.cur_angle= random.randint(5,10)
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

def call_action_line_PID(line_pos, vel):

    line_position = line_pos

    # Use PID control to calculate motor control output
    control = PID_Line_Follower.follow_line(line_position)

    # Update robot position based on control output
    action_vel = vel
    action_rot =+ control
    return action_vel, action_rot
    
def call_action_PID(xt, yt,xf,yf):
    
    distance_to_target = np.sqrt((xf - xt) ** 2 + (yf - yt) ** 2)
    if distance_to_target < 0.01:  # Stop if close enough to the target
        distance_to_target=0
        angle_to_target=0
        print("Done!")
        time.sleep(2)
        PIDControl.reset_error()
        env.reset()
        env.cur_pos=[x_inicial,0,y_inicial]
        env.cur_angle= random.randint(5,10)
        env.render(mode="top_down")

    else:
        angle_to_target = np.arctan2(yf - yt, xf - xt)
    angle_difference = angle_to_target
    angle_difference = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))  # Normalize angle
    # Calculate the error for the PID controller
    linear_error = distance_to_target * np.cos(angle_difference)
    angular_error = -angle_difference
    # Compute control signals using PID controller
    linear_velocity, angular_velocity= PIDControl.update_err(linear_error, angular_error)

    return linear_velocity, angular_velocity



# @env.unwrapped.window.event
# def on_key_press(symbol, modifiers):
#     if symbol == key.BACKSPACE:
#         print('RESET')
#         env.reset()
#         env.cur_pos=[0.2,0,0.3]
#         env.cur_anlge= random.randint(5,10)
#         env.render(mode="top_down")

def update(dt):

    

    pos_x, pos_y = env.cur_pos[0],env.cur_pos[2]
    #Navegador PID
    x_vel, rot_vel = call_action_PID(pos_x, pos_y,x_objetivo,y_objetivo)
    #Seguidor de linea
    #x_vel, rot_vel = call_action_line_PID(0.2, 0.1)
    action = np.array([abs(x_vel), rot_vel])
    #print(x_vel,rot_vel)
    obs, reward, done, info = env.step(action)
    env.render(mode="top_down")
    if done:
        print('crash!')
        PIDControl.reset_error()
        env.reset()
        env.cur_pos=[x_inicial,0,y_inicial]
        env.cur_angle= random.randint(5,10)
        env.render(mode="top_down")



pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()

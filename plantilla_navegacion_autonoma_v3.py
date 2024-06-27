#!/usr/bin/env python

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import math
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
import cv2
import time

# Configuración de argumentos para la línea de comandos
parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="Duckietown-udem1-v1")
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

# Constantes y Filtros
DUCKIE_MIN_AREA = 200  # Ajusta este valor según tus necesidades
RED_LINE_MIN_AREA = 500  # Ajusta este valor según tus necesidades
RED_COLOR = (0, 0, 255)

WHITE_FILTER_1 = np.array([0, 0, 150])
WHITE_FILTER_2 = np.array([180, 60, 255])
YELLOW_FILTER_1 = np.array([20, 50, 100])
YELLOW_FILTER_2 = np.array([30, 225, 255])

# Variables globales
last_vel = 0.44
init_time = 0
stop_count = 0

# Inicializar el entorno
if args.env_name and 'Duckietown' in args.env_name:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render(mode="top_down")

# Funciones auxiliares
def lineDetector(low_array, high_array, converted):
    mask = cv2.inRange(converted, low_array, high_array)
    segment_image = cv2.bitwise_and(converted, converted, mask=mask)
    image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)
    bordes = cv2.Canny(image, 250, 300)
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=50)

    return bordes, lineas

def calculate_control_actions(line_white, line_yellow, width):
    x_vel = 0.3  # Velocidad lineal base
    rot_vel = 0  # Velocidad angular base

    if line_white is not None:
        center_offset = line_white - width * 0.75  # Ajuste para seguir la línea interna de la línea blanca exterior
        rot_vel = -0.005 * center_offset

    # Disminuir la velocidad en los giros
    if abs(rot_vel) > 0.1:
        x_vel = 0.2

    return x_vel, rot_vel

def line_follower(obs):
    converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    height, width = converted.shape[:2]
    converted = converted[height//2:, :]

    low_yellow = np.array([20, 50, 100])
    high_yellow = np.array([30, 225, 255])
    low_white = np.array([0, 0, 150])
    high_white = np.array([180, 60, 255])

    obs_BGR = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    obs_BGR = obs_BGR[height//2:, :]

    image_y, lines_y = lineDetector(low_yellow, high_yellow, converted)
    image_w, lines_w = lineDetector(low_white, high_white, converted)

    cv2.imshow("Canny", image_y)
    cv2.imshow("Deteccion de lineas", image_w)
    cv2.waitKey(1)

    line_white = None

    if lines_w is not None:
        line_white = np.mean([l[0] for l in lines_w[:,0]])

    x_vel, rot_vel = calculate_control_actions(line_white, None, width)
    return np.array([x_vel, rot_vel])

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.cur_pos = initial_pos.copy()
        env.cur_angle = initial_angle
        env.render(mode="top_down")

key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

action = np.array([0.0, 0.0])

def update(dt):
    global action
    if key_handler[key.UP]:
        action[0] += 0.44
    if key_handler[key.DOWN]:
        action[0] -= 0.44
    if key_handler[key.LEFT]:
        action[1] += 1
    if key_handler[key.RIGHT]:
        action[1] -= 1
    if key_handler[key.SPACE]:
        action = np.array([0, 0])
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    action = line_follower(obs)

    env.render(mode="top_down")

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
pyglet.app.run()
env.close()

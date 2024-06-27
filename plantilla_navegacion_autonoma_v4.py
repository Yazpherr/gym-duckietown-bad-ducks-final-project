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

# Argumentos de línea de comandos
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

# Filtros para detectar líneas blancas y amarillas
white_filter_1 = np.array([0, 0, 0])
white_filter_2 = np.array([0, 0, 0])
yellow_filter_1 = np.array([0, 0, 0])
yellow_filter_2 = np.array([0, 0, 0])
window_filter_name = "filtro"

# Constantes
DUCKIE_MIN_AREA = 0  # Modificar si es necesario
RED_LINE_MIN_AREA = 0  # Modificar si es necesario
RED_COLOR = (0, 0, 255)
MAX_DELAY = 20
MAX_TOLERANCE = 50

# Variables globales
last_vel = 0.44
delay = -1
tolerance = -1

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

initial_pos = env.cur_pos.copy()
initial_angle = env.cur_angle

# Funciones auxiliares
def box_area(box):
    return abs(box[2][0] - box[0][0]) * abs(box[2][1] - box[0][1])

def bounding_box_height(box):
    return abs(box[2][0] - box[0][0])

def get_angle_degrees2(x1, y1, x2, y2):
    return get_angle_degrees(x1, y1, x2, y2) if y1 < y2 else get_angle_degrees(x2, y2, x1, y1)

def get_angle_degrees(x1, y1, x2, y2):
    ret_val = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
    if ret_val < 0:
        return 180.0 + ret_val
    return ret_val

def get_angle_radians(x1, y1, x2, y2):
    ret_val = math.atan2(y2 - y1, x2 - x1)
    if ret_val < 0:
        return math.pi + ret_val
    return ret_val

# Detección de intersección de líneas
def line_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    d = (by2 - by1) * (ax2 - ax1) - (bx2 - bx1) * (ay2 - ay1)
    if d:
        uA = ((bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)) / d
        uB = ((ax2 - ax1) * (ay1 - by1) - (ay2 - ay1) * (ax1 - bx1)) / d
    else:
        return None, None
    if not (0 <= uA <= 1 and 0 <= uB <= 1):
        return None, None
    x = ax1 + uA * (ax2 - ax1)
    y = ay1 + uA * (ay2 - ay1)
    return x, y

def yellow_conds(x1, y1, x2, y2):
    '''
    Condiciones para omitir el procesamiento de una línea amarilla detectada:
    si su ángulo es cercano a 0 o 180, así como si es cercano a recto.
    '''
    angle = get_angle_degrees2(x1, y1, x2, y2)
    return (angle < 30 or angle > 160) or (angle < 110 and angle > 90)

def white_conds(x1, y1, x2, y2):
    '''
    Condiciones para omitir el procesamiento de una línea blanca detectada:
    si se encuentra en el primer, segundo o tercer cuadrante, en otras palabras,
    se retorna False solo si la línea está en el cuarto cuadrante.
    '''
    return (min(x1, x2) < 320) or (min(y1, y2) < 320)

# Detección de duckies
def duckie_detection(obs, converted, frame):
    detection = False
    angle = 0
    # Implementar filtros y detección de duckies aquí
    return detection, angle

# Detección de líneas rojas
def red_line_detection(converted, frame):
    detection = False
    # Implementar filtros y detección de líneas rojas aquí
    return detection, frame

# Detección de líneas blancas y amarillas
def get_line(converted, filter_1, filter_2, line_color):
    height, width = converted.shape[:2]
    img = np.zeros_like(converted)
    img[260:640, 0:640] = converted[260:640, 0:640]
    mask = cv2.inRange(img, filter_1, filter_2)
    segment_image = cv2.bitwise_and(img, converted, mask=mask)
    image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)
    kernel = np.ones((5, 5), np.uint8)
    image_lines = cv2.erode(image, kernel, iterations=2)
    gray_lines = cv2.cvtColor(image_lines, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_lines, 50, 200, None, 3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30)
    angle = [None, None]
    line_points = (0, 0, 0, 0)

    if lines is not None:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            line_points = (x1, y1, x2, y2)
            angle = [get_angle_radians(line_points), get_angle_degrees(line_points)]
            cv2.line(image_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow(line_color, image_lines)
    return angle, line_points, image_lines

# Seguir las líneas detectadas
def line_follower(vel, angle, obs):
    converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    frame = obs[:, :, [2, 1, 0]]
    frame = cv2.UMat(frame).get()
    x_vel, rot_vel = vel, angle
    duckie_detection_, duckie_angle = duckie_detection(obs=obs, frame=frame, converted=converted)

    if duckie_detection_:
        x_vel = 0
        rot_vel += duckie_angle
        return np.array([abs(x_vel), rot_vel])

    stop_detection, red_line_frame = red_line_detection(converted=converted, frame=frame)
    global init_time
    global stop_count

    if stop_count != 0:
        stop_count -= 1

    if stop_detection and stop_count == 0:
        current_time = time.time()
        if init_time == 0:
            init_time = current_time
        if (current_time - init_time) <= 3:
            x_vel = 0
            rot_vel = 0
            return np.array([abs(x_vel), rot_vel])
        else:
            init_time = 0
            x_vel += 0.44
            stop_count = 5
            return np.array([abs(x_vel), rot_vel])

    white_filter_1 = np.array([0, 0, 150])
    white_filter_2 = np.array([180, 60, 255])
    angle_white, line_white, road1 = get_line(converted, white_filter_1, white_filter_2, "white")

    yellow_filter_1 = np.array([20, 50, 100])
    yellow_filter_2 = np.array([30, 225, 255])
    angle_yellow, line_yellow, road2 = get_line(converted, yellow_filter_1, yellow_filter_2, "yellow")

    road_frame = cv2.addWeighted(road1, 1, road2, 1, 0)
    road_detections = cv2.addWeighted(road_frame, 1, red_line_frame, 1, 0)
    all_detections = cv2.addWeighted(road_detections, 0.6, frame, 0.4, 0)
    cv2.imshow("detections", all_detections)

    if not np.all(line_white == [0, 0, 0, 0]) or not np.all(line_yellow == [0, 0, 0, 0]):
        intersect = line_intersect(*line_white, *line_yellow)
        if intersect[0] is not None:
            x_objetivo, y_objetivo = intersect
            pos_x, pos_y = env.cur_pos[0], env.cur_pos[2]
            x_vel, rot_vel = PID_controller(pos_x, pos_y, x_objetivo, y_objetivo)
        else:
            x_vel = 0
        if angle_white[0] is not None:
            if (angle_white[1] >= 70) and angle_white[0] > 30:
                print(f"GIRO ROTUNDO - BLANCA, ANGULO {round(angle_white[1], 2)}")
                rot_vel = 1
                x_vel = 0.1
            elif (angle_white[1] > 30 and angle_white[1] < 40) or (angle_white[1] < -30 and angle_white[1] > -40):
                print(f"LINEA VERTICAL - BLANCA {round(angle_white[1], 2)}")
                rot_vel = 0.2
                x_vel = 0.2
            else:
                print(f"GIRO SUAVE - BLANCA {round(angle_white[1], 2)}")
                rot_vel = -0.5
                x_vel = 0.2
        else:
            if angle_yellow[0] is not None:
                if (angle_yellow[1]) and angle_yellow[0] > 30:
                    print(f"GIRO ROTUNDO - AMARILLA {round(angle_yellow[1], 2)}")
                    rot_vel = 1
                    x_vel = 0.1
                elif (angle_yellow[1] > 30 and angle_yellow[1] < 40) or (angle_yellow[1] < -30 and angle_yellow[1] > -40):
                    print(f"LINEA VERTICAL - AMARILLA {round(angle_yellow[1], 2)}")
                    rot_vel = 0.2
                    x_vel = 0.2
                else:
                    print(f"GIRO SUAVE - AMARILLA {round(angle_yellow[1], 2)}")
                    rot_vel = 0.5
                    x_vel = 0.2
    else:
        x_vel = 0.44

    return np.array([x_vel, rot_vel])

# Manejar eventos de teclas
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
    action = line_follower(action[0], action[1], obs)
    env.render(mode="top_down")

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Iniciar el bucle principal
pyglet.app.run()

env.close()

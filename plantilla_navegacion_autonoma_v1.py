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
MAX_DELAY = 20
MAX_TOLERANCE = 50

WHITE_FILTER_1 = np.array([0, 0, 150])
WHITE_FILTER_2 = np.array([180, 60, 255])
YELLOW_FILTER_1 = np.array([20, 50, 100])
YELLOW_FILTER_2 = np.array([30, 225, 255])

# Variables globales
last_vel = 0.44
delay = -1
tolerance = -1
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
def box_area(box):
    return abs(box[2][0] - box[0][0]) * abs(box[2][1] - box[0][1])

def bounding_box_height(box):
    return abs(box[2][0] - box[0][0])

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

def line_intersect(line1, line2):
    """ returns a (x, y) tuple or None if there is no intersection """
    ax1, ay1, ax2, ay2 = line1
    bx1, by1, bx2, by2 = line2
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

def get_line(converted, filter_1, filter_2, line_color):
    """
    Obtiene el ángulo y los puntos de una línea detectada de un color específico.

    :param converted: Imagen convertida en formato HSV
    :param filter_1: Primer filtro para la detección de color
    :param filter_2: Segundo filtro para la detección de color
    :param line_color: Color de la línea a detectar ("white" o "yellow")
    :return: Ángulo, puntos de la línea y la imagen con las líneas detectadas
    """
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
    line_points = [0, 0, 0, 0]

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_points = [x1, y1, x2, y2]
            angle = [get_angle_radians(x1, y1, x2, y2), get_angle_degrees(x1, y1, x2, y2)]
            cv2.line(image_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
            break  # only consider the first line for simplicity

    cv2.imshow(line_color, image_lines)
    return angle, line_points, image_lines

def duckie_detection_function(obs, converted, frame):
    """
    Función para detectar duckies en la imagen.
    """
    # Se asume que no hay detección
    detection = False
    angle = 0

    # Definir los límites de color para la detección de duckies en el espacio de color HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Crear una máscara que capture todas las áreas amarillas
    yellow_mask = cv2.inRange(converted, lower_yellow, upper_yellow)

    # Aplicar operaciones morfológicas para eliminar ruido
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.erode(yellow_mask, kernel, iterations=2)
    yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=2)

    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos en la imagen original
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > DUCKIE_MIN_AREA:
            detection = True
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Calcular el ángulo de evasión basado en la posición del duckie
            if x + w / 2 < frame.shape[1] / 2:
                angle = -0.5  # Evitar hacia la izquierda
            else:
                angle = 0.5  # Evitar hacia la derecha

    # Mostrar la máscara de detección (opcional)
    cv2.imshow("Duckie detection", yellow_mask)

    return detection, angle

def red_line_detection(converted, frame):
    """
    Función para detectar líneas rojas en el camino.
    """
    # Se asume que no hay detección
    detection = False

    # Definir los límites de color para la detección de líneas rojas en el espacio de color HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Crear una máscara que capture todas las áreas rojas
    mask1 = cv2.inRange(converted, lower_red1, upper_red1)
    mask2 = cv2.inRange(converted, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Aplicar operaciones morfológicas para eliminar ruido
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.erode(red_mask, kernel, iterations=2)
    red_mask = cv2.dilate(red_mask, kernel, iterations=2)

    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos en la imagen original
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > RED_LINE_MIN_AREA:
            detection = True
            cv2.drawContours(frame, [contour], -1, RED_COLOR, 3)

    # Mostrar la máscara de detección (opcional)
    cv2.imshow("Red line detection", red_mask)

    return detection, frame

def PID_controller(pos_x, pos_y, x_objetivo, y_objetivo):
    """
    Controlador PID para determinar la velocidad lineal y angular.
    """
    # Esta es una implementación de muestra y debe ser ajustada
    k_p = 1.0  # Ganancia proporcional
    error_x = x_objetivo - pos_x
    error_y = y_objetivo - pos_y
    x_vel = k_p * error_x
    rot_vel = k_p * error_y
    return x_vel, rot_vel

def line_follower(vel, angle, obs):
    """
    Controlador principal para seguir líneas y evitar obstáculos.

    :param vel: Velocidad actual
    :param angle: Ángulo actual
    :param obs: Observaciones actuales del entorno
    :return: Nueva velocidad y ángulo
    """
    converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    frame = obs[:, :, [2, 1, 0]]
    frame = cv2.UMat(frame).get()

    global init_time
    global stop_count

    duckie_detection, duckie_angle = duckie_detection_function(obs, converted, frame)
    if duckie_detection:
        return np.array([0, duckie_angle])

    stop_detection, red_line_frame = red_line_detection(converted, frame)
    if stop_detection and stop_count == 0:
        current_time = time.time()
        if init_time == 0:
            init_time = current_time
        if (current_time - init_time) <= 3:
            return np.array([0, 0])
        else:
            init_time = 0
            stop_count = 5
            return np.array([0.44, 0])

    if stop_count != 0:
        stop_count -= 1

    angle_white, line_white, road1 = get_line(converted, WHITE_FILTER_1, WHITE_FILTER_2, "white")
    angle_yellow, line_yellow, road2 = get_line(converted, YELLOW_FILTER_1, YELLOW_FILTER_2, "yellow")

    road_frame = cv2.addWeighted(road1, 1, road2, 1, 0)
    all_detections = cv2.addWeighted(road_frame, 0.6, frame, 0.4, 0)
    cv2.imshow("detections", all_detections)

    x_vel = 0  # Definir x_vel y rot_vel por defecto
    rot_vel = 0

    if line_white != [0, 0, 0, 0] or line_yellow != [0, 0, 0, 0]:
        intersect = line_intersect(line_white, line_yellow)
        if intersect[0] is not None:
            x_objetivo, y_objetivo = intersect
            pos_x, pos_y = env.cur_pos[0], env.cur_pos[2]
            x_vel, rot_vel = PID_controller(pos_x, pos_y, x_objetivo, y_objetivo)

        if angle_white[0] is not None:
            if angle_white[1] >= 70 or angle_white[1] <= -70 and angle_white[0] > 40:
                print(f"GIRO ROTUNDO - BLANCA, ANGULO {round(angle_white[1], 2)}")
                rot_vel = 1
                x_vel = 0.1
            elif 30 < angle_white[1] < 40 or -40 < angle_white[1] < -30:
                print(f"LINEA VERTICAL - BLANCA {round(angle_white[1], 2)}")
                rot_vel = 0.2
                x_vel = 0.2
            else:
                print(f"GIRO SUAVE - BLANCA {round(angle_white[1], 2)}")
                rot_vel = 0.25
                x_vel = 0.3

        elif angle_yellow[0] is not None:
            if angle_yellow[1] >= 70 or angle_yellow[1] <= -70 and angle_yellow[0] > 40:
                print(f"GIRO ROTUNDO - AMARILLA {round(angle_yellow[1], 2)}")
                rot_vel = 1
                x_vel = 0.1
            elif 30 < angle_yellow[1] < 40 or -40 < angle_yellow[1] < -30:
                print(f"LINEA VERTICAL - AMARILLA {round(angle_yellow[1], 2)}")
                rot_vel = 0.2
                x_vel = 0.2
            else:
                print(f"GIRO SUAVE - AMARILLA {round(angle_yellow[1], 2)}")
                rot_vel = 0.45
                x_vel = 0.3
    else:
        x_vel = 0.44

    return np.array([x_vel, rot_vel])

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    Maneja las pulsaciones de teclas para reiniciar el entorno.
    """
    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.cur_pos = [x_inicial, 0, y_inicial]
        env.cur_angle = init_angle
        env.render(mode="top_down")

key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

action = np.array([0.0, 0.0])

def update(dt):
    """
    Función que se llama en cada paso de tiempo para actualizar el estado del robot.
    """
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
    vel, angle = line_follower(action[0], action[1], obs)
    action[0] = vel
    action[1] = angle

    env.render(mode="top_down")

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
pyglet.app.run()
env.close()

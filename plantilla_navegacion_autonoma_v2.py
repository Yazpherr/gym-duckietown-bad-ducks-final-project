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
def lineDetector(low_array, high_array, color, converted, obs_BGR):
    # Filtrar colores de la imagen en el rango utilizando 
    mask = cv2.inRange(converted, low_array, high_array)
    # Bitwise-AND mask and original
    segment_image = cv2.bitwise_and(converted, converted, mask=mask)
    # Vuelta al espacio BGR
    image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)
    # Se añade detector de bordes canny
    bordes = cv2.Canny(image, 250, 300)
    # Se añade detector de lineas Hough 
    lineas = cv2.HoughLines(bordes, 1, np.pi / 180, 100)

    # Condicional para que no crashee al no encontrar lineas
    if type(lineas) == np.ndarray:
        for linea in lineas:
            rho, theta = linea[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(obs_BGR, (x1, y1), (x2, y2), color, 2)
    return bordes

def calculate_control_actions(line_white, line_yellow):
    """
    Calcula las acciones de control para mantener el robot en el centro del carril.
    """
    x_vel = 0.3  # Velocidad lineal base
    rot_vel = 0  # Velocidad angular base

    if line_white is not None and line_yellow is not None:
        center_offset = (line_white + line_yellow) / 2 - 320  # La imagen tiene un ancho de 640 píxeles
        rot_vel = -0.005 * center_offset  # Ajuste proporcional simple
    elif line_white is not None:
        center_offset = line_white - 320
        rot_vel = -0.005 * center_offset
    elif line_yellow is not None:
        center_offset = line_yellow - 320
        rot_vel = -0.005 * center_offset

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

    low_yellow = np.array([20, 50, 100])
    high_yellow = np.array([30, 225, 255])
    low_white = np.array([0, 0, 150])
    high_white = np.array([180, 60, 255])

    obs_BGR = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    image_y = lineDetector(low_yellow, high_yellow, (255, 0, 0), converted, obs_BGR)
    image_w = lineDetector(low_white, high_white, (0, 0, 255), converted, obs_BGR)

    image_yw = cv2.bitwise_or(image_y, image_w)
    bordes_resize = cv2.resize(image_yw, (480, 360))
    obs_resize = cv2.resize(obs_BGR, (480, 360))

    cv2.imshow("Canny", bordes_resize)
    cv2.imshow("Deteccion de lineas", obs_resize)
    cv2.waitKey(1)

    # Obtener el centro de las líneas detectadas
    line_white = None
    line_yellow = None

    if image_w is not None:
        M = cv2.moments(image_w)
        if M["m00"] != 0:
            line_white = int(M["m10"] / M["m00"])
    
    if image_y is not None:
        M = cv2.moments(image_y)
        if M["m00"] != 0:
            line_yellow = int(M["m10"] / M["m00"])

    x_vel, rot_vel = calculate_control_actions(line_white, line_yellow)
    return np.array([x_vel, rot_vel])

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

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    Maneja las pulsaciones de teclas para reiniciar el entorno.
    """
    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.cur_pos = initial_pos.copy()
        env.cur_angle = initial_angle
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

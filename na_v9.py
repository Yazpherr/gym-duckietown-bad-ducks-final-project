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
import cv2
import time

# Argumentos para configuración del entorno
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

# CONSTANTES
RED_LINE_MIN_AREA = 50000
RED_COLOR = (0, 0, 255)
LINES_COLOR = {"white": (255, 255, 255), "yellow": (0, 215, 255)}

# Variables para detección en rojo
init_time = 0
stop_count = 0

# PID CONTROLLER
Kp = 0.1
Kd = 0.10
Ki = 0.5

x_inicial = 0.90
y_inicial = 3.30
init_angle = 0

# Clase para el controlador PID
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.previous_error = 0

    def update(self, error):
        self.integral += error
        derivative = error - self.previous_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output

    def reset(self):
        self.integral = 0
        self.previous_error = 0

# Crear instancia del entorno Duckietown
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
    pid_velocity = PIDController(Kp, Ki, Kd)
    pid_rotation = PIDController(Kp, Ki, Kd)
else:
    env = gym.make(args.env_name)

# Resetear y configurar el entorno
env.reset()
env.cur_pos = [x_inicial, 0, y_inicial]
env.cur_angle = init_angle
env.render(mode="top_down")

# Filtros para detección de líneas
white_filter_1 = np.array([0, 0, 150])
white_filter_2 = np.array([180, 60, 255])

yellow_filter_1 = np.array([20, 50, 100])
yellow_filter_2 = np.array([30, 225, 255])

# Función para obtener el ángulo en grados entre dos puntos
def get_angle_degrees(x1, y1, x2, y2):
    ret_val = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
    if ret_val < 0:
        return 180.0 + ret_val
    return ret_val

# Función para obtener el ángulo en radianes entre dos puntos
def get_angle_radians(x1, y1, x2, y2):
    ret_val = math.atan2(y2 - y1, x2 - x1)
    if ret_val < 0:
        return math.pi + ret_val
    return ret_val

# Función para obtener la línea detectada y su ángulo usando Transformada de Hough
def get_line(converted, filter_1, filter_2, line_color):
    img = np.zeros_like(converted)
    img[260:640, 0:640] = converted[260:640, 0:640]
    mask = cv2.inRange(img, filter_1, filter_2)
    segment_image = cv2.bitwise_and(img, converted, mask=mask)

    image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)
    kernel = np.ones((5, 5), np.uint8)
    image_lines = cv2.erode(image, kernel, iterations=2)

    gray_lines = cv2.cvtColor(image_lines, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_lines, 150, 300, None, 3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 30)
    angle = [None, None]
    line_points = (0, 0, 0, 0)

    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            line_points = (x1, y1, x2, y2)
            angle = [get_angle_radians(x1, y1, x2, y2), get_angle_degrees(x1, y1, x2, y2)]
            cv2.line(image_lines, (x1, y1), (x2, y2), LINES_COLOR[line_color], 2)
    
    cv2.imshow(f"Detected Lines - {line_color}", image_lines)
    cv2.waitKey(1)  # Para refrescar la ventana

    return angle, line_points, image_lines

# Función para la detección de patos
def duckie_detection(obs, converted, frame):
    detection = False
    angle = 0

    filtro_1 = np.array([15, 240, 100])
    filtro_2 = np.array([30, 255, 255])

    mask_duckie = cv2.inRange(converted, filtro_1, filtro_2)
    kernel = np.ones((5, 5), np.uint8)

    image_out = cv2.erode(mask_duckie, kernel, iterations=2)
    image_out = cv2.dilate(image_out, kernel, iterations=10)

    segment_image = cv2.bitwise_and(converted, converted, mask=mask_duckie)

    contours, _ = cv2.findContours(image_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segment_image_post_opening = cv2.bitwise_and(converted, converted, mask=image_out)
    segment_image_post_opening = cv2.cvtColor(segment_image_post_opening, cv2.COLOR_HSV2BGR)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        duckie_box_area = w * h

        if duckie_box_area > 5000 and h > 150:  # Ajustar para detección desde más lejos
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 3)
            cv2.putText(frame, "Pato detectado", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            detection = True
            angle = 1

    return detection, angle, frame

# Función para la detección de líneas rojas
def red_line_detection(converted, frame):
    detection = False

    height, width = converted.shape[:2]
    img = np.zeros_like(converted)
    img[260:640, 0:640] = converted[260:640, 0:640]

    red_filter1 = np.array([175, 100, 20])
    red_filter2 = np.array([179, 255, 255])

    mask_line = cv2.inRange(img, red_filter1, red_filter2)
    kernel = np.ones((5, 5), np.uint8)

    image_out = cv2.erode(mask_line, kernel, iterations=2)
    image_out = cv2.dilate(mask_line, kernel, iterations=2)

    segment_image = cv2.bitwise_and(img, img, mask=mask_line)
    contours, _ = cv2.findContours(image_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        red_line_area = w * h

        if red_line_area > RED_LINE_MIN_AREA:
            x2 = x + w
            y2 = y + h

            cv2.rectangle(segment_image, (int(x), int(y)), (int(x2), int(y2)), RED_COLOR, 3)

            if red_line_area > 50000 and (h > 80 and w > 520):
                detection = True

    cv2.imshow("Red Line Detection", segment_image)
    cv2.waitKey(1)  # Para refrescar la ventana

    return detection, segment_image

# Función mejorada para el controlador PID
def compute_pid_control(xt, yt, xf, yf):
    distance_to_target = np.sqrt((xf - xt) ** 2 + (yf - yt) ** 2)
    angle_difference = np.arctan2(yf - yt, xf - xt)
    angle_difference = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))  # Normalizar

    linear_error = distance_to_target * np.cos(angle_difference)
    angular_error = -angle_difference

    linear_velocity = pid_velocity.update(linear_error)
    angular_velocity = pid_rotation.update(angular_error)

    return linear_velocity, angular_velocity

# Función principal para seguir líneas y evitar obstáculos
def line_follower(vel, angle, obs):
    global init_time
    global stop_count

    # Convertir la observación a formato HSV y BGR
    converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    
    x_vel, rot_vel = vel, angle

    # Detección de patos
    duckie_detected, duckie_angle, duckie_frame = duckie_detection(obs=obs, frame=frame, converted=converted)

    if duckie_detected:
        x_vel = 0
        rot_vel += duckie_angle
        cv2.imshow("detections", duckie_frame)
        return np.array([abs(x_vel), rot_vel])

    # Detección de líneas rojas
    stop_detected, red_line_frame = red_line_detection(converted=converted, frame=frame)

    if stop_count != 0:
        stop_count -= 1

    if stop_detected and stop_count == 0:
        current_time = time.time()
        if init_time == 0:
            init_time = current_time
        if (current_time - init_time) <= 3:
            x_vel = 0
            rot_vel = 0
            all_detections = cv2.addWeighted(duckie_frame, 0.4, red_line_frame, 0.6, 0)
            cv2.imshow("detections", all_detections)
            return np.array([abs(x_vel), rot_vel])
        else:
            init_time = 0
            x_vel += 0.44
            stop_count = 5
            all_detections = cv2.addWeighted(duckie_frame, 0.4, red_line_frame, 0.6, 0)
            cv2.imshow("detections", all_detections)
            return np.array([abs(x_vel), rot_vel])

    # Detección de líneas blancas y amarillas
    angle_white, line_white, road1 = get_line(converted, white_filter_1, white_filter_2, "white")
    angle_yellow, line_yellow, road2 = get_line(converted, yellow_filter_1, yellow_filter_2, "yellow")

    road_frame = cv2.addWeighted(road1, 1, road2, 1, 0)
    road_detections = cv2.addWeighted(road_frame, 1, red_line_frame, 1, 0)
    all_detections = cv2.addWeighted(road_detections, 0.6, duckie_frame, 0.4, 0)

    cv2.imshow("detections", all_detections)

    # Asegurarse de que el Duckiebot esté en el carril derecho
    if line_white != (0, 0, 0, 0) and line_yellow != (0, 0, 0, 0):
        mid_x = (line_white[0] + line_yellow[0]) / 2
        if env.cur_pos[0] < mid_x:
            rot_vel += 0.5
        else:
            rot_vel -= 0.5
    else:
        x_vel = 0

    # Ajustar la velocidad y rotación basada en las detecciones de líneas
    if angle_white[0] is not None:
        if angle_white[1] >= 70 and angle_white[0] > 30:
            rot_vel = 1
            x_vel = 0.1
        elif 30 < angle_white[1] < 40 or -40 < angle_white[1] < -30:
            rot_vel = 0
            x_vel = 0.2
        else:
            rot_vel = -0.5
            x_vel = 0.2
            
    elif angle_yellow[0] is not None:
        if angle_yellow[1] >= 70 and angle_yellow[0] > 30:
            rot_vel = -1
            x_vel = 0.1
        elif 30 < angle_yellow[1] < 40 or -40 < angle_yellow[1] < -30:
            rot_vel = 0
            x_vel = 0.2
        else:
            rot_vel = 0.5
            x_vel = 0.2
    else:
        x_vel = 0.44

    return np.array([abs(x_vel), rot_vel])

# Función para manejar eventos de teclado
@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render(mode="top_down")

    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0

    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

action = np.array([0.44, 0.0])

# Función de actualización que se llama en cada frame
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
    vel, angle = line_follower(action[0], action[1], obs)
    action[0] = vel
    action[1] = angle

    env.render(mode="top_down")

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

pyglet.app.run()

env.close()

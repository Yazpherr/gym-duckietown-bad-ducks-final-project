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

# ===============================
# Configuración de argumentos
# ===============================
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

# ===============================
# Definición de constantes
# ===============================
DUCKIE_MIN_AREA = 0
RED_LINE_MIN_AREA = 50000
RED_COLOR = (0, 0, 255)
LINES_COLOR = {"white": (255, 255, 255), "yellow": (0, 215, 255)}

WHITE_FILTER_1 = np.array([0, 0, 150])
WHITE_FILTER_2 = np.array([180, 60, 255])
YELLOW_FILTER_1 = np.array([20, 50, 100])
YELLOW_FILTER_2 = np.array([30, 225, 255])

# Variables para detección en rojo
init_time = 0
stop_count = 0

# ===============================
# Controlador PID
# ===============================
# Constantes para el controlador PID
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
        self.integral_vel = 0
        self.previous_error_vel = 0
        self.integral_rot = 0
        self.previous_error_rot = 0

    def update_error(self, error_vel, error_rot):
        self.integral_vel += error_vel
        self.integral_rot += error_rot
        derivative_vel = error_vel - self.previous_error_vel
        derivative_rot = error_rot - self.previous_error_rot
        control_vel = self.Kp * error_vel + self.Ki * self.integral_vel + self.Kd * derivative_vel
        control_rot = self.Kp * error_rot + self.Ki * self.integral_rot + self.Kd * derivative_rot
        self.previous_error_vel = error_vel
        self.previous_error_rot = error_rot
        return control_vel, control_rot

    def reset_error(self):
        self.previous_error_vel = 0
        self.integral_vel = 0
        self.previous_error_rot = 0
        self.integral_rot = 0

# ===============================
# Inicialización del entorno
# ===============================
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
    PIDControl = PIDController(Kp, Ki, Kd)
else:
    env = gym.make(args.env_name)

# Resetear y configurar el entorno
env.reset()
env.cur_pos = [x_inicial, 0, y_inicial]
env.cur_angle = init_angle
env.render(mode="top_down")

# ===============================
# Definición de filtros de color
# ===============================
# Filtros para detección de líneas
white_filter_1 = np.array([0, 0, 150])
white_filter_2 = np.array([180, 60, 255])

yellow_filter_1 = np.array([20, 50, 100])
yellow_filter_2 = np.array([30, 225, 255])

# ===============================
# Funciones auxiliares
# ===============================
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

# Función para encontrar la intersección entre dos líneas
def line_intersect(line1, line2):
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

# Función para la detección de líneas usando Canny y HoughLinesP
def lineDetector(low_array, high_array, converted):
    mask = cv2.inRange(converted, low_array, high_array)
    segment_image = cv2.bitwise_and(converted, converted, mask=mask)
    image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)
    bordes = cv2.Canny(image, 250, 300)
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=50)

    return bordes, lineas

# Función para calcular las acciones de control basadas en la detección de líneas
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

# ===============================
# Funciones de detección de líneas y obstáculos
# ===============================
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
    edges = cv2.Canny(gray_lines, 50, 200, None, 3)

    # Usar Transformada de Hough para detectar líneas
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
    
    # Mostrar la ventana con la imagen procesada
    cv2.imshow(f"Detected Lines - {line_color}", image_lines)
    cv2.waitKey(1)  # Para refrescar la ventana

    return angle, line_points, image_lines

# Función para detección de patos
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
    image_out = cv2.dilate(image_out, kernel, iterations=2)

    segment_image = cv2.bitwise_and(img, img, mask=mask_line)
    contours, _ = cv2.findContours(image_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        red_line_area = w * h

        if red_line_area > RED_LINE_MIN_AREA:
            x2 = x + w
            y2 = y + h

            cv2.rectangle(segment_image, (int(x), int(y)), (int(x2), int(y2)), RED_COLOR, 3)
            print("RED AREA: H, W", red_line_area, h, w)

            if red_line_area > 50000 and (h > 80 and w > 520):
                detection = True

    # Mostrar la ventana con la imagen procesada
    cv2.imshow("Red Line Detection", segment_image)
    cv2.waitKey(1)  # Para refrescar la ventana

    return detection, segment_image

# ===============================
# Controlador PID para seguir la línea
# ===============================
# Función para el controlador PID
def PID_controller(xt, yt, xf, yf):
    distance_to_target = np.sqrt((xf - xt) ** 2 + (yf - yt) ** 2)
    angle_difference = np.arctan2(yf - yt, xf - xt)
    angle_difference = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))  # Normalizar

    linear_error = distance_to_target * np.cos(angle_difference)
    angular_error = -angle_difference

    linear_velocity, angular_velocity = PIDControl.update_error(linear_error, angular_error)

    return linear_velocity, angular_velocity

# ===============================
# Función principal de seguimiento de línea
# ===============================
def line_follower(obs):
    global init_time
    global stop_count

    converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    height, width = converted.shape[:2]
    converted = converted[height//2:, :]

    obs_BGR = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    obs_BGR = obs_BGR[height//2:, :]

    # Detección de líneas amarillas y blancas usando el nuevo algoritmo
    image_y, lines_y = lineDetector(YELLOW_FILTER_1, YELLOW_FILTER_2, converted)
    image_w, lines_w = lineDetector(WHITE_FILTER_1, WHITE_FILTER_2, converted)

    cv2.imshow("Canny - Yellow", image_y)
    cv2.imshow("Canny - White", image_w)
    cv2.waitKey(1)

    line_white = None
    line_yellow = None

    if lines_w is not None:
        line_white = np.mean([l[0] for l in lines_w[:, 0]])

    if lines_y is not None:
        line_yellow = np.mean([l[0] for l in lines_y[:, 0]])

    x_vel, rot_vel = calculate_control_actions(line_white, line_yellow, width)

    # Detección de líneas rojas
    stop_detected, red_line_frame = red_line_detection(converted=converted, frame=obs_BGR)

    if stop_detected:
        if stop_count == 0:
            stop_count = 1
            init_time = time.time()
            x_vel = 0
            rot_vel = 0
        elif time.time() - init_time >= 2:
            stop_count = 0
            x_vel = 0.44  # Decidir avanzar o girar
            # Implementar lógica para decidir si avanzar o girar
        else:
            x_vel = 0
            rot_vel = 0

        cv2.imshow("detections", red_line_frame)
        return np.array([x_vel, rot_vel])

    # Implementar condiciones de giro rotundo, giro suave y línea vertical
    if line_white is not None:
        if isinstance(line_white, np.ndarray) and line_white.size > 1:
            if 70 <= abs(line_white[1]) < 90 and line_white[0] > 30:
                print(f"GIRO ROTUNDO - BLANCA, ANGULO {round(line_white[1], 2)}")
                rot_vel = 1
                x_vel = 0.1
            elif 30 < abs(line_white[1]) < 40 or -40 < line_white[1] < -30:
                print(f"LINEA VERTICAL - BLANCA {round(line_white[1], 2)}")
                rot_vel = 0.2
                x_vel = 0.2
            else:
                print(f"GIRO SUAVE - BLANCA {round(line_white[1], 2)}")
                rot_vel = -0.5
                x_vel = 0.2
    elif line_yellow is not None:
        if isinstance(line_yellow, np.ndarray) and line_yellow.size > 1:
            if 70 <= abs(line_yellow[1]) < 90 and line_yellow[0] > 30:
                print(f"GIRO ROTUNDO - AMARILLA, ANGULO {round(line_yellow[1], 2)}")
                rot_vel = -1  # Ajustar para girar hacia el carril derecho
                x_vel = 0.1
            elif 30 < abs(line_yellow[1]) < 40 or -40 < line_yellow[1] < -30:
                print(f"LINEA VERTICAL - AMARILLA {round(line_yellow[1], 2)}")
                rot_vel = -0.2  # Ajustar para mantener el carril derecho
                x_vel = 0.2
            else:
                print(f"GIRO SUAVE - AMARILLA {round(line_yellow[1], 2)}")
                rot_vel = 0.5
                x_vel = 0.2

    return np.array([x_vel, rot_vel])

# ===============================
# Manejo de eventos de teclado
# ===============================
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

# ===============================
# Función de actualización
# ===============================
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
    action = line_follower(obs)

    env.render(mode="top_down")

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

pyglet.app.run()

env.close()

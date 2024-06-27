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
DUCKIE_MIN_AREA  = 0
RED_LINE_MIN_AREA = 50000
RED_COLOR = (0,0,255)
LINES_COLOR = {"white": (255, 255, 255), "yellow": (0, 215, 255)}

# Variables para detección en rojo
init_time = 0
stop_count = 0

# PID CONTROLLER________________________________________________________________________________________
Kp = 0.1
Kd = 0.15
Ki = 0.12

x_inicial = 1.2
y_inicial = 0.0

init_angle = 0

# PID CONTROLLER________________________________________________________________________________________
Kp = 0.1
Kd = 0.15
Ki = 0.12

x_inicial = 1.2
y_inicial = 0.0

init_angle = 0

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

    def update_error(self, error_vel, error_rot):
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
        self.previous_error = 0
        self.integral = 0
    
    def PID_controller(xt, yt, xf, yf):
        distance_to_target = np.sqrt((xf - xt) ** 2 + (yf - yt) ** 2)
        angle_difference = np.arctan2(yf - yt, xf - xt)
        angle_difference = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))  # Normalizar

        # Calcular el error del controlador PID
        linear_error = distance_to_target * np.cos(angle_difference)
        angular_error = -angle_difference

        # Calcular controles usando PID
        linear_velocity, angular_velocity = PIDControl.update_error(linear_error, angular_error)

        return linear_velocity, angular_velocity
#...............
    if args.env_name and args.env_name.find('Duckietown') != -1:
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

#----------------------------------------------
    # env.reset()
    env.cur_pos = [x_inicial, 0, y_inicial]  # Posición inicial de duckiebot
    env.cur_angle = init_angle  # Ángulo inicial de duckiebot
    env.render(mode="top_down")
    
      
    
# Parametros para el detector de lineas blancas
white_filter_1 = np.array([0, 0, 0])
white_filter_2 = np.array([0, 0, 0])

# Filtros para el detector de lineas amarillas
yellow_filter_1 = np.array([0, 0, 0])
yellow_filter_2 = np.array([0, 0, 0])
window_filter_name = "filtro"

# Constantes
DUCKIE_MIN_AREA = 0 #editar esto si es necesario
RED_LINE_MIN_AREA = 0 #editar esto si es necesario
RED_COLOR = (0,0,255)
MAX_DELAY = 20
MAX_TOLERANCE = 50

# Variables globales
last_vel = 0.44
delay = -1
tolerance = -1

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
    )
    PIDControl = PIDController[Kp,Ki,Kd]
else:
    env = gym.make(args.env_name)

env.reset()
env.render(mode="top_down")


# Funciones interesantes para hacer operaciones interesantes
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


def line_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """ returns a (x, y) tuple or None if there is no intersection """
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
    return (min(x1,x2) < 320) or (min(y1,y2) < 320)



def duckie_detection(obs, converted, frame):
    '''
    Detectar patos, retornar si hubo detección y el ángulo de giro en tal caso 
    para lograr esquivar el duckie y evitar la colisión.
    '''
    # Se asume que no hay detección
    detection = False
    angle = 0

    # Implementar filtros
    filtro_1 = np.array([15, 240, 100])  # Rango mínimo para amarillo/naranja en HSV
    filtro_2 = np.array([30, 255, 255])  # Rango máximo para amarillo/naranja en HSV

    mask_duckie = cv2.inRange(converted, filtro_1, filtro_2)  # Aplicar máscara para filtrar duckie

    # Segmentación con operaciones morfológicas (erode y dilate)
    kernel = np.ones((5, 5), np.uint8)  # Crear kernel para operaciones morfológicas

    image_out = cv2.erode(mask_duckie, kernel, iterations=2)  # Operación morfológica erode
    image_out = cv2.dilate(image_out, kernel, iterations=10)  # Operación morfológica dilate

    segment_image = cv2.bitwise_and(converted, converted, mask=mask_duckie)  # Aplicar segmentación

    contours, hierarchy = cv2.findContours(image_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Observar la imagen post-opening
    segment_image_post_opening = cv2.bitwise_and(converted, converted, mask=image_out)
    segment_image_post_opening = cv2.cvtColor(segment_image_post_opening, cv2.COLOR_HSV2BGR)

    # Revisar los contornos identificados y dibujar el rectángulo correspondiente
    for cnt in contours:
        # Obtener rectángulo
        x, y, w, h = cv2.boundingRect(cnt)
        duckie_box_area = w * h

        if duckie_box_area > 7500 and h > 300:
            cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 3)
            detection = True
            angle = 1

    # # Se asume que no hay detección
    # detection = False
    # angle = 0

    # '''
    # Para lograr la detección, se puede utilizar lo realizado en el desafío 1
    # con el freno de emergencia, aunque con la diferencia que ya no será un freno,
    # sino que será un método creado por ustedes para lograr esquivar al duckie.
    # '''

    # # Implementar filtros
    # white_filter_1 = np.array([0, 0, 150])
    # white_filter_2 = np.array([180, 60, 255])

    # angle_white, line_white, road1 = get_line(converted, white_filter_1, white_filter_2, "white")

    # # Filtros para el detector de líneas amarillas
    # yellow_filter_1 = np.array([20, 50, 100])
    # yellow_filter_2 = np.array([30, 225, 255])

    # angle_yellow, line_yellow, road2 = get_line(converted, yellow_filter_1, yellow_filter_2, "yellow")


    # # Realizar la segmentación con operaciones morfológicas (erode y dilate) 
    # # y buscar los contornos


    # # Revisar los contornos identificados y dibujar el rectángulo correspondiente
    # # a la detección, además, dentro de este for, se establece la detección = verdadera
    # # además del ángulo de giro angle = 'ángulo'


    #         # if duckie_box_area > 7500 and h > 300:
    #         #     cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 3)
    #         #     detection = True
    #         #     angle = 1



    # # Mostrar ventanas con los resultados
    # #cv2.imshow("Patos filtro", segment_image_post_opening)
    # #cv2.imshow("Patos detecciones", frame)

    # return detection, angle, frame 



def red_line_detection(converted, frame):
    '''
    Detección de líneas rojas en el camino, esto es análogo a la detección de duckies,
    pero con otros filtros, notar también, que no es necesario aplicar houghlines en este caso
    '''
    # Se asume que no hay detección
    detection = False

    # Cortar la imagen para obtener la zona relevante
    height, width = converted.shape[:2]
    img = np.zeros_like(converted)
    img[260:640, 0:640] = converted[260:640, 0:640]

    # Obtener e implementar filtros
    red_filter1 = np.array([175, 100, 20])   # Rango mínimo para rojo en HSV
    red_filter2 = np.array([179, 255, 255])  # Rango máximo para rojo en HSV

    mask_line = cv2.inRange(img, red_filter1, red_filter2)  # Aplicar máscara para filtro

    # Segmentación con operaciones morfológicas (erosión y dilatación)
    kernel = np.ones((5, 5), np.uint8)  # Crear kernel para operaciones morfológicas

    image_out = cv2.erode(mask_line, kernel, iterations=2)  # Operación morfológica de erosión
    image_out = cv2.dilate(image_out, kernel, iterations=2)  # Operación morfológica de dilatación

    # Realizar la segmentación con operaciones morfológicas (erode y dilate)
    # y buscar los contornos
    segment_image = cv2.bitwise_and(img, img, mask=mask_line)  # Aplicar segmentació
    
    contours, hierarchy = cv2.findContours(image_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Revisar los contornos identificados y dibujar el rectángulo correspondiente
    # a la detección, además, Si hay detección, detection = True
    # Revisar los contornos identificados y dibujar el rectángulo correspondiente
    for cnt in contours:
        # Obtener Rectángulo
        x, y, w, h = cv2.boundingRect(cnt)
        red_line_area = w * h

        # Filtrar por área mínima
        if red_line_area > RED_LINE_MIN_AREA:
            x2 = x + w
            y2 = y + h

            # Dibujar un rectángulo en la imagen
            cv2.rectangle(segment_image, (int(x), int(y)), (int(x2), int(y2)), RED_COLOR, 3)
            print("RED AREA: H, W", red_line_area, h, w)
  
            if red_line_area > 50000 and (h > 80 and w > 520):
                detection = True
    

   # cv2.imshow("Red Lines", segment_image)

   # Mostrar ventanas con los resultados
    return detection, segment_image 



def get_line(converted, filter_1, filter_2, line_color):
    '''
    Determina el ángulo al que debe girar el duckiebot dependiendo
    del filtro aplicado, y de qué color trata, si es "white"
    y se cumplen las condiciones entonces gira a la izquierda,
    si es "yellow" y se cumplen las condiciones gira a la derecha.
    '''
    # Cortar la imagen para obtener la zona relevante
    height, width = converted.shape[:2]
    img = np.zeros_like(converted)
    img[260:640, 0:640] = converted[260:640, 0:640]
    
    mask = cv2.inRange(img, filter_1, filter_2)
    segment_image = cv2.bitwise_and(img, converted, mask=mask)

    # Erosionar la imagen
    image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)
    kernel = np.ones((5, 5), np.uint8)
    image_lines = cv2.erode(image, kernel, iterations=2)

    # Detectar líneas
    gray_lines = cv2.cvtColor(image_lines, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_lines, 50, 200, None, 3)

    # Detectar líneas usando HoughLinesP
    lines = cv2.HoughLines(edges, 1, np.pi/180, 30)
    angle = [None, None]
    line_points = (0, 0, 0, 0)

    # Si encontramos una línea
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Coordenadas del inicio y coordenadas finales de las líneas
                line_points = (x1, y1, x2, y2)
                # Obtenemos el ángulo de la línea como lista
                angle = [get_angle_radians(line_points),
                         get_angle_degrees(line_points)]
                # Dibujamos la línea en la imagen
                cv2.line(image_lines, (x1, y1), (x2, y2), line_color, 2)

    cv2.imshow(line_color, image_lines)

    # Se cubre cada color por separado, tanto el amarillo como el blanco
    # Con esto, ya se puede determinar mediante condiciones el movimiento del giro del robot.
    # Por ejemplo, si tiene una línea blanca muy cercana a la derecha, debe doblar hacia la izquierda
    # y viceversa.
    return angle, line_points, image_lines



def line_follower(vel, angle, obs):
    converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    frame = obs[:, :, [2, 1, 0]]
    frame = cv2.UMat(frame).get()

    # Detección de duckies
    x_vel, rot_vel = vel, angle


    '''
    Implementar evasión de duckies en el camino, variado la velocidad angular del robot
    '''

    duckie_detection_, duckie_angle, duckie_frame= duckie_detection(obs=obs, frame=frame, converted=converted)

    # Tratamos de esquivar el duckie
    if duckie_detection_:
        x_vel = 0
        rot_vel += duckie_angle
        return np.array((abs(x_vel), rot_vel))



    # Detección de líneas rojas
    stop_detection, red_line_frame = red_line_detection(converted=converted, frame=frame)
    '''
    Implementar detención por un tiempo determinado del duckiebot
    al detectar una linea roja en el camino, luego de este tiempo,
    el duckiebot debe seguir avanzando
    '''
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


    # Obtencion de lineas blancas
    # Filtros para el detector de líneas blancas
    white_filter_1 = np.array([0, 0, 150])
    white_filter_2 = np.array([180, 60, 255])

    angle_white, line_white, road1 = get_line(converted, white_filter_1, white_filter_2, "white")

    # Filtros para el detector de líneas amarillas
    yellow_filter_1 = np.array([20, 50, 100])
    yellow_filter_2 = np.array([30, 225, 255])

    angle_yellow, line_yellow, road2 = get_line(converted, yellow_filter_1, yellow_filter_2, "yellow")
   
    # MUESTRO MIS DETECCIONES
    road_frame = cv2.addWeighted(road1, 1, road2, 1, 0)
    road_detections = cv2.addWeighted(road_frame, 1, red_line_frame, 1, 0)
    all_detections = cv2.addWeighted(road_detections, 0.6, duckie_frame, 0.4, 0)

    cv2.imshow("detections", all_detections)  # mostrar todas las detecciones



    # # '''
    # # Implementar un controlador para poder navegar dentro del mapa con los ángulos obtenidos
    # # en las líneas anteriores
    # # '''
    if line_white != (0, 0, 0, 0) or line_yellow != (0, 0, 0, 0):
        
        intersect = line_intersect(line_white, line_yellow)

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
                rot_vel = 0
                x_vel = 0.2

            else:  # Si está inclinado en otro sentido, giro suave
                print(f"GIRO SUAVE - BLANCA {round(angle_white[1], 2)}")
                rot_vel = -0.5
                x_vel = 0.2
        else: 
           # if angle_yellow[0] is not None:
            if (angle_yellow[1] ) and angle_yellow[0] > 30:
                print(f"GIRO ROTUNDO - AMARILLA {round(angle_yellow[1], 2)}")
                rot_vel = 1
                x_vel = 0.1

            elif (angle_yellow[1] > 30 and angle_yellow[1] < 40) or (angle_yellow[1] < -30 and angle_yellow[1] > -40):
                print(f"LINEA VERTICAL - AMARILLA {round(angle_yellow[1], 2)}")
                rot_vel = 0
                x_vel = 0.2

            else:  # Si está inclinado en otro sentido, giro suave
                print(f"GIRO SUAVE - AMARILLA {round(angle_yellow[1], 2)}")
                rot_vel = -0.5
                x_vel = 0.2
    # cuando no detecto lineas quiero ir derecho 
    else: 
        x_vel = 0.44

    vel, angle = np.angle([abs(x_vel), rot_vel])

    return np.array([vel, angle])

    #return np.array([vel, 'new_angle']) # Implementar nuevo ángulo de giro controlado







    # ''' 
    # Implementar detención por un tiempo determinado del duckiebot
    # al detectar una linea roja en el camino, luego de este tiempo,
    # el duckiebot debe seguir avanzando
    # '''
    # global init_time
    # global stop_count

    # if stop_count != 0:
    #     stop_count -= 1


    # if stop_detection and stop_count == 0:
    #         current_time = time.time()

    #         if init_time == 0:
    #             init_time = current_time
    #         if (current_time - init_time) <= 3:
    #             x_vel = 0
    #             rot_vel = 0
    #             return np.array([abs(x_vel), rot_vel])
    #         else:
    #             init_time = 0
    #             x_vel += 0.44
    #             stop_count = 5
    #             return np.array([abs(x_vel), rot_vel])

    # # Obtener el ángulo propuesto por cada color
    # _, angle_white = get_line(converted, white_filter_1, white_filter_2, "white")
    # _, angle_yellow = get_line(converted, yellow_filter_1, yellow_filter_2, "yellow")



    
    # return np.array([vel, 'new_angle']) # Implementar nuevo ángulo de giro controlado



# Definir variables globales
init_time = 0
stop_count = 0

import time

def line_follower(vel, angle, obs):
    global init_time
    global stop_count
    
    converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    frame = obs[:, :, [2, 1, 0]]
    frame = cv2.UMat(frame).get()

    # Detección de duckies
    x_vel, rot_vel = vel, angle

    duckie_detection_, duckie_angle, duckie_frame = duckie_detection(obs=obs, frame=frame, converted=converted)

    # Tratamos de esquivar el duckie
    if duckie_detection_:
        x_vel = 0                       # velocidad lineal 
        rot_vel += duckie_angle         # velocidad rotacional 
        return np.array([abs(x_vel), rot_vel])

    # Detección de líneas rojas
    stop_detection, red_line_frame = red_line_detection(converted=converted, frame=frame)

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

    # Filtros para el detector de líneas blancas
    white_filter_1 = np.array([0, 0, 150])
    white_filter_2 = np.array([180, 60, 255])
    angle_white, line_white, road1 = get_line(converted, white_filter_1, white_filter_2, "white")

    # Filtros para el detector de líneas amarillas
    yellow_filter_1 = np.array([20, 50, 100])
    yellow_filter_2 = np.array([30, 225, 255])
    angle_yellow, line_yellow, road2 = get_line(converted, yellow_filter_1, yellow_filter_2, "yellow")

    # MUESTRO MIS DETECCIONES
    road_frame = cv2.addWeighted(road1, 1, road2, 1, 0)
    road_detections = cv2.addWeighted(road_frame, 1, red_line_frame, 1, 0)
    all_detections = cv2.addWeighted(road_detections, 0.6, duckie_frame, 0.4, 0)

    cv2.imshow("detections", all_detections)  # mostrar todas las detecciones

    if line_white != [0, 0, 0, 0] or line_yellow != [0, 0, 0, 0]:
        intersect = line_intersect(line_white, line_yellow)

        if intersect[0] is not None:
            x_objetivo, y_objetivo = intersect

            pos_x, pos_y = env.cur_pos[0], env.cur_pos[2]
            x_vel, rot_vel = PID_controller(pos_x, pos_y, x_objetivo, y_objetivo)

        else:
            x_vel = 0

        if angle_white[0] is not None:
            if angle_white[1] >= 70 and angle_white[0] > 30:
                print(f"GIRO ROTUNDO - BLANCA, ANGULO {round(angle_white[1], 2)}")
                rot_vel = 1
                x_vel = 0.1

            elif 30 < angle_white[1] < 40 or -40 < angle_white[1] < -30:
                print(f"LINEA VERTICAL - BLANCA {round(angle_white[1], 2)}")
                rot_vel = 0.2
                x_vel = 0.2

            else:  # Si está inclinado en otro sentido, giro suave
                print(f"GIRO SUAVE - BLANCA {round(angle_white[1], 2)}")
                rot_vel = -0.5
                x_vel = 0.2
        else:
            if angle_yellow[1] and angle_yellow[0] > 30:
                print(f"GIRO ROTUNDO - AMARILLA {round(angle_yellow[1], 2)}")
                rot_vel = 1
                x_vel = 0.1

            elif 30 < angle_yellow[1] < 40 or -40 < angle_yellow[1] < -30:
                print(f"LINEA VERTICAL - AMARILLA {round(angle_yellow[1], 2)}")
                rot_vel = 0.2
                x_vel = 0.2

            else:  # Si está inclinado en otro sentido, giro suave
                print(f"GIRO SUAVE - AMARILLA {round(angle_yellow[1], 2)}")
                rot_vel = 0.5
                x_vel = 0.2
    else: 
        x_vel = 0.44

    vel, angle = np.array([abs(x_vel), rot_vel])

    return np.array([vel, angle])




@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    Handler para reiniciar el ambiente
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render(mode="top_down")

    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0

    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)


# Registrar el handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

action = np.array([0.44, 0.0])

def update(dt):
    """
    Funcion que se llama en step.
    """
    global action
    # Aquí se controla el duckiebot
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

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    ''' Aquí se obtienen las observaciones y se setea la acción
    Para esto, se debe utilizar la función creada anteriormente llamada line_follower,
    la cual recibe como argumentos la velocidad lineal, la velocidad angular y 
    la ventana de la visualización, en este caso obs.
    Luego, se setea la acción del movimiento implementado con el controlador
    con action[i], donde i es 0 y 1, (lineal y angular)
    '''


    # obs consiste en un imagen de 640 x 480 x 3
    obs, reward, done, info = env.step(action)
    vel, angle = line_follower(action[0], action[1], obs)
    action[0] = vel
    action[1] = angle

    # if done:
    #     print('done!')
    #     env.reset()
    #     env.render(mode="top_down")

    env.render(mode="top_down")

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
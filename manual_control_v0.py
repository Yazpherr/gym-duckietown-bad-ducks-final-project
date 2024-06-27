#!/usr/bin/env python

"""
Este script permite visualizar la vista superior del simulador Duckiebot y seleccionar dos puntos en el mapa.
"""
import cv2  # Importación del módulo 'cv2' de OpenCV para procesamiento de imágenes.
import gym_duckietown  # Importación del módulo 'gym_duckietown' para entornos Duckietown.
from gym_duckietown.envs import DuckietownEnv  # Importación de la clase 'DuckietownEnv'.

# Configuración del entorno
env = DuckietownEnv(
    seed=1,  # Semilla aleatoria.
    map_name='udem1',  # Nombre del mapa.
    draw_curve=False,  # No dibujar la curva de seguimiento del carril.
    draw_bbox=False,  # No dibujar cajas de delimitación de detección de colisiones.
    domain_rand=False,  # No habilitar la aleatorización de dominio.
    frame_skip=1,  # Número de fotogramas para saltar.
    distortion=False,  # No aplicar distorsión de la cámara.
)

env.reset()  # Reinicio del entorno.

# Variables para las posiciones inicial y final
start_pos = None
goal_pos = None
points_selected = []

# Función de callback para manejar los eventos del mouse
def mouse_callback(event, x, y, flags, param):
    global start_pos, goal_pos, points_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_selected) < 2:
            # Convertir la posición de pixel en coordenadas del entorno
            pos = (x / env.window_width * env.grid_width, y / env.window_height * env.grid_height)
            points_selected.append(pos)
            print(f"Posición seleccionada: {pos}")
            if len(points_selected) == 1:
                start_pos = pos
                print(f"Posición inicial establecida en: {start_pos}")
            elif len(points_selected) == 2:
                goal_pos = pos
                print(f"Posición final establecida en: {goal_pos}")

# Asignar la función de callback a la ventana de OpenCV
cv2.namedWindow("Top View")
cv2.setMouseCallback("Top View", mouse_callback)

# Función para capturar la vista superior
def get_top_view():
    try:
        top_view = env.render(mode='top_down')
        if top_view is None:
            raise ValueError("La imagen capturada de la vista superior está vacía.")
        return top_view
    except Exception as e:
        print(f"Error al capturar la vista superior: {e}")
        return None

# Bucle principal para mostrar la vista superior
while True:
    env.step([0, 0])  # Añadimos esta línea para actualizar el entorno
    top_view = get_top_view()
    if top_view is not None:
        # Dibujar los puntos seleccionados
        for point in points_selected:
            cv2.circle(top_view, (int(point[0] * env.window_width / env.grid_width), int(point[1] * env.window_height / env.grid_height)), 5, (0, 0, 255), -1)
        cv2.imshow("Top View", top_view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

env.close()
cv2.destroyAllWindows()

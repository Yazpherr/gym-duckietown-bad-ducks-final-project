#!/usr/bin/env python

"""
Este script permite visualizar la vista superior del simulador Duckiebot.
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

# Función para capturar la vista superior
def get_top_view():
    try:
        top_view = env.render(mode='top_down')
        return top_view
    except Exception as e:
        print(f"Error al capturar la vista superior: {e}")
        return None

# Bucle principal para mostrar la vista superior
while True:
    top_view = get_top_view()
    if top_view is not None:
        cv2.imshow("Top View", top_view)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

env.close()
cv2.destroyAllWindows()

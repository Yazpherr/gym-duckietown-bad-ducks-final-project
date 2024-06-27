#!/usr/bin/env python

"""
Este script permite controlar manualmente el simulador Duckiebot usando las teclas de flecha del teclado y visualizar la vista superior.
"""
import cv2  # Importación del módulo 'cv2' de OpenCV para procesamiento de imágenes.
import gym_duckietown  # Importación del módulo 'gym_duckietown' para entornos Duckietown.
from gym_duckietown.envs import DuckietownEnv  # Importación de la clase 'DuckietownEnv'.
from pyglet.window import key  # Importación del módulo 'key' de pyglet para manejar eventos de teclado.
import numpy as np  # Importación del módulo 'numpy' para operaciones numéricas.
import pyglet  # Importación del módulo 'pyglet' para crear aplicaciones multimedia en Python.

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
env.render()  # Representación gráfica del entorno.

# Manejador de eventos de teclado
@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    Este controlador procesa los comandos del teclado que controlan la simulación.
    """
    if symbol == key.BACKSPACE or symbol == key.SLASH:  # Si se presiona la tecla Retroceso o Slash.
        print('RESET')  # Imprimir mensaje de reinicio.
        env.reset()  # Reiniciar el entorno.
        env.render()  # Representar gráficamente el entorno.
    elif symbol == key.ESCAPE:  # Si se presiona la tecla Escape.
        env.close()  # Cerrar el entorno.
        cv2.destroyAllWindows()
        sys.exit(0)  # Salir del programa.

# Registrar un manejador de teclado
key_handler = key.KeyStateHandler()  # Creación de un manejador de teclado.
env.unwrapped.window.push_handlers(key_handler)  # Asociación del manejador de teclado con la ventana del entorno.

# Función para capturar la vista superior
def get_top_view():
    try:
        top_view = env.render(mode='top_down')
        return top_view
    except Exception as e:
        print(f"Error al capturar la vista superior: {e}")
        return None

# Función para actualizar y procesar los fotogramas del entorno
def update(dt):
    action = np.array([0.0, 0.0])  # Inicialización de la acción del agente.

    if key_handler[key.UP]:  # Si se presiona la tecla de flecha hacia arriba.
        action = np.array([0.44, 0.0])  # Movimiento hacia adelante.
    if key_handler[key.DOWN]:  # Si se presiona la tecla de flecha hacia abajo.
        action = np.array([-0.44, 0])  # Movimiento hacia atrás.
    if key_handler[key.LEFT]:  # Si se presiona la tecla de flecha hacia la izquierda.
        action = np.array([0.35, +1])  # Giro a la izquierda.
    if key_handler[key.RIGHT]:  # Si se presiona la tecla de flecha hacia la derecha.
        action = np.array([0.35, -1])  # Giro a la derecha.
    if key_handler[key.SPACE]:  # Si se presiona la tecla Espacio.
        action = np.array([0, 0])  # Detenerse.
    if key_handler[key.LSHIFT]:  # Si se presiona la tecla Mayúsculas Izquierda.
        action *= 1.5  # Aumentar la velocidad.

    obs, reward, done, info = env.step(action)  # Ejecutar la acción en el entorno y obtener la observación, recompensa, estado y información.
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))  # Imprimir el número de pasos y la recompensa.

    if done:  # Si el episodio ha terminado.
        print('¡hecho!')  # Imprimir mensaje de finalización.
        env.reset()  # Reiniciar el entorno.
        env.render()  # Representar gráficamente el entorno.

    # Mostrar la vista superior
    top_view = get_top_view()
    if top_view is not None:
        cv2.imshow("Top View", top_view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            env.close()
            cv2.destroyAllWindows()
            sys.exit(0)  # Salir del programa

# Programar la actualización de los fotogramas
pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)  # Programar la actualización de los fotogramas.

# Entrar en el bucle principal de eventos
pyglet.app.run()  # Ejecutar la aplicación de pyglet.
env.close()  # Cerrar el entorno al finalizar.

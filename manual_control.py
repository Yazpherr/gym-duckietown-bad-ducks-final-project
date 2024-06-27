#!/usr/bin/env python
# manual

"""
Este script te permite controlar manualmente el simulador o Duckiebot
utilizando las teclas de flecha del teclado.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper

# Definir los argumentos de línea de comandos
parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown_udem1')
parser.add_argument('--map-name', default='proyecto_final_patomalo')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='dibujar la curva de seguimiento del carril')
parser.add_argument('--draw-bbox', action='store_true', help='dibujar las cajas delimitadoras de detección de colisiones')
parser.add_argument('--domain-rand', action='store_true', help='habilitar la randomización de dominio')
parser.add_argument('--frame-skip', default=1, type=int, help='número de fotogramas para omitir')
parser.add_argument('--seed', default=1, type=int, help='semilla')
args = parser.parse_args()

# Crear el entorno
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
else:
    env = gym.make(args.env_name)

# Reiniciar el entorno y renderizar
env.reset()
env.render()

# Manejar eventos de teclado
@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    Este manejador procesa los comandos del teclado que
    controlan la simulación.
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('REINICIAR')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Tomar una captura de pantalla
    # DESCOMENTAR SI ES NECESARIO - Dependencia de Skimage
    # elif symbol == key.RETURN:
    #     print('guardando captura de pantalla')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Registrar un manejador de teclado
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

# Actualizar el entorno en cada fotograma
def update(dt):
    """
    Esta función se llama en cada fotograma para manejar
    el movimiento y redibujar.
    """

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Aumento de velocidad
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    print('conteo_de_pasos = %s, recompensa=%.3f' % (env.unwrapped.step_count, reward))

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)

        im.save('pantalla.png')

    if done:
        print('¡hecho!')
        env.reset()
        env.render()

    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Entrar en el bucle principal de eventos
pyglet.app.run()

env.close()

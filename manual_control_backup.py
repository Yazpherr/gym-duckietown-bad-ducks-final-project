#!/usr/bin/env python
# manual

"""
Este script permite controlar manualmente el simulador o Duckiebot
usando las teclas de flecha del teclado.
"""
import sys  # Importación del módulo 'sys' para interactuar con el sistema operativo.
import argparse  # Importación del módulo 'argparse' para analizar argumentos de línea de comandos.
import pyglet  # Importación del módulo 'pyglet' para crear aplicaciones multimedia en Python.
import cv2  # Importación del módulo 'cv2' de OpenCV para procesamiento de imágenes.
from pyglet.window import key  # Importación del módulo 'key' de pyglet para manejar eventos de teclado.
import numpy as np  # Importación del módulo 'numpy' para operaciones numéricas.
import gym  # Importación del módulo 'gym' para entornos de aprendizaje por refuerzo.
import gym_duckietown  # Importación del módulo 'gym_duckietown' para entornos Duckietown.
from gym_duckietown.envs import DuckietownEnv  # Importación de la clase 'DuckietownEnv'.
from gym_duckietown.wrappers import UndistortWrapper  # Importación de la clase 'UndistortWrapper'.

# Argumentos de línea de comandos
parser = argparse.ArgumentParser()  # Creación de un objeto ArgumentParser.
parser.add_argument('--env-name', default='Duckietown_udem1')  # Argumento para el nombre del entorno.
parser.add_argument('--map-name', default='udem1')  # Argumento para el nombre del mapa.
parser.add_argument('--distortion', default=False, action='store_true')  # Argumento para la distorsión de la cámara.
parser.add_argument('--draw-curve', action='store_true', help='dibujar la curva de seguimiento del carril')  # Argumento para dibujar la curva de seguimiento del carril.
parser.add_argument('--draw-bbox', action='store_true', help='dibujar cajas de delimitación de detección de colisiones')  # Argumento para dibujar las cajas de delimitación de detección de colisiones.
parser.add_argument('--domain-rand', action='store_true', help='habilitar la aleatorización de dominio')  # Argumento para habilitar la aleatorización de dominio.
parser.add_argument('--frame-skip', default=1, type=int, help='número de fotogramas para saltar')  # Argumento para el número de fotogramas a omitir.
parser.add_argument('--seed', default=1, type=int, help='semilla')  # Argumento para la semilla del generador de números aleatorios.
args = parser.parse_args()  # Análisis de los argumentos de la línea de comandos.

# Configuración del entorno
if args.env_name and args.env_name.find('Duckietown') != -1:  # Si el nombre del entorno contiene 'Duckietown'.
    env = DuckietownEnv(  # Creación de un entorno Duckietown.
        seed=args.seed,  # Semilla aleatoria.
        map_name=args.map_name,  # Nombre del mapa.
        draw_curve=args.draw_curve,  # Dibujar la curva de seguimiento del carril.
        draw_bbox=args.draw_bbox,  # Dibujar cajas de delimitación de detección de colisiones.
        domain_rand=args.domain_rand,  # Habilitar la aleatorización de dominio.
        frame_skip=args.frame_skip,  # Número de fotogramas para saltar.
        distortion=args.distortion,  # Distorsión de la cámara.
    )
else:  # Si no es un entorno Duckietown.
    env = gym.make(args.env_name)  # Creación de un entorno Gym.

env.reset()  # Reinicio del entorno.
env.render()  # Representación gráfica del entorno.

# Función para controlar los valores de los trackbars
def nothing(x):
    pass

# Crear una ventana
cv2.namedWindow('image')  # Creación de una ventana con el nombre 'image'.

# Crear trackbars para cambio de color
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)  # Trackbar para el mínimo de Hue.
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)  # Trackbar para el máximo de Hue.

cv2.createTrackbar('SMin', 'image', 0, 255, nothing)  # Trackbar para el mínimo de Saturación.
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)  # Trackbar para el máximo de Saturación.

cv2.createTrackbar('VMin', 'image', 0, 255, nothing)  # Trackbar para el mínimo de Valor.
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)  # Trackbar para el máximo de Valor.

# Establecer valores predeterminados para los trackbars HSV máximos
cv2.setTrackbarPos('HMax', 'image', 179)  # Hue máximo inicializado en 179.
cv2.setTrackbarPos('SMax', 'image', 255)  # Saturación máximo inicializado en 255.
cv2.setTrackbarPos('VMax', 'image', 255)  # Valor máximo inicializado en 255.

# Inicializar para verificar si cambia el valor HSV mínimo/máximo
hMin = sMin = vMin = hMax = sMax = vMax = 0  # Inicialización de los valores HSV mínimos y máximos.
phMin = psMin = pvMin = phMax = psMax = pvMax = 0  # Inicialización de los valores anteriores de HSV.

# Función para aplicar un efecto de alerta roja sobre el fotograma
def red_alert(frame):
    red_img = np.zeros((480, 640, 3), dtype=np.uint8)  # Creación de una imagen roja.
    red_img[:, :, 2] = 90  # Establecimiento del canal de rojo.
    blend = cv2.addWeighted(frame, 0.5, red_img, 0.5, 0)  # Mezcla de la imagen original y la imagen roja.
    return blend

# Manejador de eventos de teclado
@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    Este controlador procesa los comandos del teclado que
    controlan la simulación
    """
    if symbol == key.BACKSPACE or symbol == key.SLASH:  # Si se presiona la tecla Retroceso o Slash.
        print('RESET')  # Imprimir mensaje de reinicio.
        env.reset()  # Reiniciar el entorno.
        env.render()  # Representar gráficamente el entorno.
    elif symbol == key.PAGEUP:  # Si se presiona la tecla Página Arriba.
        env.unwrapped.cam_angle[0] = 0  # Establecer el ángulo de la cámara en cero.
    elif symbol == key.ESCAPE:  # Si se presiona la tecla Escape.
        env.close()  # Cerrar el entorno.
        sys.exit(0)  # Salir del programa.

# Registrar un manejador de teclado
key_handler = key.KeyStateHandler()  # Creación de un manejador de teclado.
env.unwrapped.window.push_handlers(key_handler)  # Asociación del manejador de teclado con la ventana del entorno.


# Variable para controlar el estado de la alerta roja
boleano = False
# Función para actualizar y procesar los fotogramas del entorno
# def update(dt):
#     global hMin, sMin, vMin, hMax, sMax, vMax  # Variables globales para los valores HSV mínimos y máximos.
#     global phMin, psMin, pvMin, phMax, psMax, pvMax  # Variables globales para los valores HSV anteriores.

#     action = np.array([0.0, 0.0])  # Inicialización de la acción del agente.

#     if key_handler[key.UP]:  # Si se presiona la tecla de flecha hacia arriba.
#         action = np.array([0.44, 0.0])  # Movimiento hacia adelante.
def update(dt):
    global hMin, sMin, vMin, hMax, sMax, vMax  # Variables globales para los valores HSV mínimos y máximos.
    global phMin, psMin, pvMin, phMax, psMax, pvMax  # Variables globales para los valores HSV anteriores.
    global boleano  # Variable global para controlar el estado de la alerta roja.

    action = np.array([0.0, 0.0])  # Inicialización de la acción del agente.

    if key_handler[key.UP] and not boleano:  # Si se presiona la tecla de flecha hacia arriba y no hay alerta roja.
        action = np.array([0.44, 0.0])  # Movimiento hacia adelante.
    elif key_handler[key.UP] and boleano:  # Si se presiona la tecla de flecha hacia arriba y hay alerta roja.
        action = np.array([-0.1, 0])  # Freno de emergencia. retrocede 

    if key_handler[key.DOWN]:  # Si se presiona la tecla de flecha hacia abajo.
        action = np.array([-0.44, 0])  # Movimiento hacia atrás.

    if key_handler[key.LEFT] and not boleano:  # Si se presiona la tecla de flecha hacia izquierda y no hay alerta roja.
        action = np.array([0.35, +1])  # Movimiento hacia adelante.
    elif key_handler[key.LEFT] and boleano:  # Si se presiona la tecla de flecha hacia izquierda y hay alerta roja.
        action = np.array([-0.1, 0])  # Freno de emergencia. retrocede 

    if key_handler[key.RIGHT] and not boleano:  # Si se presiona la tecla de flecha hacia derecha y no hay alerta roja.
        action = np.array([0.35, -1])  # Movimiento hacia adelante.
    elif key_handler[key.RIGHT] and boleano:  # Si se presiona la tecla de flecha hacia derecha y hay alerta roja.
        action = np.array([-0.1, 0])  # Freno de emergencia. retrocede 


    # if key_handler[key.LEFT]:  # Si se presiona la tecla de flecha hacia la izquierda.
    #     action = np.array([0.35, +1])  # Giro a la izquierda.
    # if key_handler[key.RIGHT]:  # Si se presiona la tecla de flecha hacia la derecha.
    #     action = np.array([0.35, -1])  # Giro a la derecha.
    
    
    
    if key_handler[key.SPACE]:  # Si se presiona la tecla Espacio.
        action = np.array([0, 0])  # Detenerse.

    # Aumento de velocidad
    if key_handler[key.LSHIFT]:  # Si se presiona la tecla Mayúsculas Izquierda.
        action *= 1.5  # Aumentar la velocidad.

    obs, reward, done, info = env.step(action)  # Ejecutar la acción en el entorno y obtener la observación, recompensa, estado y información.
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)  # Convertir la observación a RGB.
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))  # Imprimir el número de pasos y la recompensa.

    if key_handler[key.RETURN]:  # Si se presiona la tecla Enter.
        from PIL import Image  # Importar la clase Image de PIL.
        im = Image.fromarray(obs)  # Crear una imagen a partir de la observación.
        im.save('screen.png')  # Guardar la imagen como 'screen.png'.

    if done:  # Si el episodio ha terminado.
        print('¡hecho!')  # Imprimir mensaje de finalización.
        env.render()  # Representar gráficamente el entorno.

    hMin = cv2.getTrackbarPos('HMin', 'image')  # Obtener el valor del trackbar de HMin.
    sMin = cv2.getTrackbarPos('SMin', 'image')  # Obtener el valor del trackbar de SMin.
    vMin = cv2.getTrackbarPos('VMin', 'image')  # Obtener el valor del trackbar de VMin.

    hMax = cv2.getTrackbarPos('HMax', 'image')  # Obtener el valor del trackbar de HMax.
    sMax = cv2.getTrackbarPos('SMax', 'image')  # Obtener el valor del trackbar de SMax.
    vMax = cv2.getTrackbarPos('VMax', 'image')  # Obtener el valor del trackbar de VMax.

    lower = np.array([90, 250, 133])  # Crear un array con los valores mínimos de HSV.
    upper = np.array([179, 255, 255])  # Crear un array con los valores máximos de HSV.

    hsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)  # Convertir la observación a espacio de color HSV.
    mask = cv2.inRange(hsv, lower, upper)  # Crear una máscara utilizando los valores de rango. 

    # Una vez identificado apliciones operaciones morfologicas
    # Definir el kernel para la operación de erosión
    kernel = np.ones((5,5), np.uint8)  # Puedes ajustar el tamaño del kernel según tus necesidades
    maskEroded = cv2.erode(mask, kernel, iterations=2)# le aplico la operacion morfolofia de dilatación
    maskDilated = cv2.dilate(maskEroded, kernel, iterations=10) # le aplico la operacion morfolofia de dilatación
    maskPostOps = maskDilated

        # Encontrar contornos en la imagen binarizada
    contours, hierarchy = cv2.findContours(maskPostOps, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar bounding boxes alrededor de los contornos de los objetos detectados (en este caso, los patos)
    # for contour in contours:
    #     # Calcular el bounding box para el contorno actual
    #     x, y, w, h = cv2.boundingRect(contour)
    #     # Dibujar el bounding box sobre la imagen original (obs)
    #     cv2.rectangle(obs, (x, y), (x + w, y + h), (0,0, 255), 2)  # Color rojo, grosor 2

    # Definir el umbral para activar la alerta roja (por ejemplo, el área mínima del bounding box)
    umbral_alerta = 40000  # Puedes ajustar este valor según sea necesario

    # Variable para controlar el estado de la alerta roja
    boleano = False

    # Dibujar bounding boxes alrededor de los contornos de los objetos detectados (en este caso, los patos)
    for contour in contours:
        # Calcular el bounding box para el contorno actual
        x, y, w, h = cv2.boundingRect(contour)
        # Dibujar el bounding box sobre la imagen original (obs)
        cv2.rectangle(obs, (x, y), (x + w, y + h), (0,0, 255), 2)  # Color rojo, grosor 2
        # Calcular el área del bounding box
        area_bbox = w * h
        
        # Si el área del bounding box supera el umbral
        if area_bbox > umbral_alerta:
            # Activar la alerta roja
            boleano = True
            texto = "Cuidado con el Cuak!"
            cv2.putText(obs, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            obs = red_alert(obs)

            # Dibujar el bounding box en color rojo
            cv2.rectangle(obs, (x, y), (x + w, y + h), (0, 0, 255), 2)

        else:
            # Dibujar el bounding box en color verde
            cv2.rectangle(obs, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Si no se detectan patos que superen el umbral, desactivar la alerta roja
    if not contours:
        boleano = False


    output = cv2.bitwise_and(obs, obs, mask=mask)  # Aplicar la máscara a la observación. <--
    

    # Mostrar la imagen con los bounding boxes
    # aquí se obtienen las observaciones y se setea la acción
    # obs consiste en un imagen de 640 x 480 x 3
    obs, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))
    
    # if done:
    #     print('done!')
    #     env.reset()
    #     env.render()

    """
    Esta parte del código convierte el espacio de color de la observación (obs) de 
    BGR a RGB, y luego convierte la imagen RGB resultante en un objeto UMat, que es 
    un tipo de dato utilizado por OpenCV para operaciones de procesamiento de imágenes 
    eficientes.
    """
    # Detección lineas
    obs = obs.astype(np.uint8)
    frame = obs[:, :, [2, 1, 0]]
    frame = cv2.UMat(frame).get()
    
    """
    Esta parte del código define una función llamada eliminate_half que toma una 
    imagen como entrada y devuelve una nueva imagen que contiene solo la mitad 
    inferior de la imagen original.
    """
    def eliminate_half(image):
        height, width = image.shape[:2]
        # Create a new array of zeros with the same shape as the original image
        new_image = np.zeros_like(image)
        # Copy the relevant half of the original image into the new array
        new_image[height//2:height, :] = image[height//2:height, :]
        return new_image
     #Eliminar mitad superior de la imagen
    converted = eliminate_half(obs)

    """
    Define cuatro matrices numpy que contienen los 
    valores de color mínimo y máximo para los colores amarillo(lineas centrales) y 
    blanco(lineas de orilla) en el espacio de color HSV. 
    Estos valores se utilizan para segmentar los colores amarillo y blanco de una imagen.
    """
    #Cambiar tipo de color de RGB a HSV
    converted = cv2.cvtColor(converted, cv2.COLOR_RGB2HSV)
    obs_BGR = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    
    low_yellow = np.array ([0, 70, 155])
    high_yellow = np.array([30, 240, 243])
    low_white = np.array([0, 0, 140])
    high_white = np.array([170, 40, 255])


    """ 
    La función filtra los colores de una imagen en un rango especificado por 
    low_array y high_array, y luego aplica el detector de bordes Canny y 
    el detector de líneas Hough para detectar líneas de color en la imagen.
    """
    #Funcion que filtra colores, aplica canny y detecta líneas de color en base a canales BGR
    def lineDetector(low_array, high_array, color):
        #Filtrar colores de la imagen en el rango utilizando 
        mask = cv2.inRange(converted, low_array, high_array)
        # Bitwise-AND mask and original
        segment_image = cv2.bitwise_and(converted, converted, mask= mask)
        #Vuelta al espacio BGR
        image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)
        #Se añade detector de bordes canny
        bordes = cv2.Canny(image, 250, 300)
        #Se añade detector de lineas Hough 
        lineas = cv2.HoughLines(bordes,1,np.pi/180,100)

        """ 
        Estas líneas de código dibujan las líneas detectadas en la imagen original 
        utilizando la función cv2.line. La función toma cinco argumentos: 
        la imagen de entrada, las coordenadas de inicio y final de la línea, 
        el color de la línea y el grosor de la línea. La función dibuja una 
        línea en la imagen de entrada utilizando las coordenadas de inicio y 
        final especificadas.
        """
       #Condicional para que no crashee al no encontrar lineas
        if type(lineas) == np.ndarray:
            for linea in lineas:
                rho,theta = linea[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(obs_BGR,(x1,y1),(x2,y2),color,2)
        return bordes

    """ 
    Estas líneas de código definen dos variables image_y y image_w que almacenan
    las imágenes de salida del detector de líneas para los colores amarillo y 
    blanco, respectivamente. La función lineDetector toma tres argumentos: 
    los valores de color mínimo y máximo para el color especificado y el color 
    de la línea a dibujar en la imagen de salida.
    """
    image_y = lineDetector(low_yellow, high_yellow, (255, 0, 0))
    image_w = lineDetector(low_white, high_white, (0, 0, 255))
    
    """ 
    A continuación, la función cv2.bitwise_or se utiliza para combinar las 
    dos imágenes de salida del detector de líneas en una sola imagen. La función
    toma dos imágenes como argumentos y devuelve una nueva imagen que contiene 
    los píxeles de ambas imágenes.
    """
    #Se juntan ambas imágenes filtradas
    image_yw = cv2.bitwise_or(image_y, image_w)
    bordes_resize = cv2.resize(image_yw, (480, 360))
    """ 
    Finalmente, la función cv2.resize se utiliza para cambiar el tamaño 
    de la imagen de observación a una resolución de 480x360 píxeles.
    """
    obs_resize = cv2.resize(obs_BGR, (480, 360))



    cv2.imshow("Canny", bordes_resize)
    cv2.imshow("Deteccion de lineas", obs_resize)
    # cv2.imshow('Mascara', mask)
    # cv2.imshow('Operaciones Morfologicas', mask)
    # cv2.imshow('Bounding Boxes', output)
    # cv2.imshow('image hsv', hsv)  # Mostrar la imagen HSV con la máscara aplicada.
    # cv2.imshow('image', output)  # Mostrar la imagen con la máscara aplicada.
    cv2.waitKey(1)  # Esperar una tecla presionada (1ms).
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    env.render()  # Representar gráficamente el entorno.
# Programar la actualización de los fotogramas
pyglet.clock.schedule_interval(update, 1.0 /  env.unwrapped.frame_rate)  # Programar la actualización de los fotogramas.
# Entrar en el bucle principal de eventos
pyglet.app.run()  # Ejecutar la aplicación de pyglet.
env.close()  # Cerrar el entorno al finalizar.



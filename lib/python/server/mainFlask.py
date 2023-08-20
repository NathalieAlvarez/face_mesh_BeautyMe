# importaciones
# pip install flask-socketio==5.2.0
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

# importación de librerías
import cv2
import mediapipe as mp
import base64
from PIL import Image
import io
import numpy as np

# inicializaoms le servidor web
app = Flask(__name__, static_folder="./templates/static")
# inicializar la conexión con websockets
app.config["SECRET_KEY"] = "secret"
socketio = SocketIO(app)


# helper function that will convert a base64-encoded image to a NumPy array that can be processed using OpenCV.
def base64_to_image(base64_string):
    # falta perfeccionar esta parte
    imgdata = base64.b64decode(base64_string)
    # se crea una imagen en la ubicaclion filename
    filename = "some_image.jpg"
    with open(filename, "wb") as f:
        f.write(imgdata)

    # imageCV2 guarda los datos previamente generados en un formato que cv puede leer
    imageCV2 = cv2.imread("some_image.jpg")
    return imageCV2


# procesa los bytes recibidos desde flutter
def process_camera_image(image_bytes):
    return Image.open(io.BytesIO(image_bytes))


def procesar_imagen(RGBA_image):
    # Convertir Uint8List a una matriz NumPy
    np_array = np.array(RGBA_image, dtype=np.uint8)

    # Asegurarse de que la matriz tiene la forma correcta (ancho, alto, canales)
    # En este caso, asumimos que la imagen es de 4 canales (RGBA)
    height, width, channels = (int(len(RGBA_image) / 4), 4, 1)
    np_array = np_array.reshape(height, width, channels)

    return np_array


# recibe una lista anidada co los 3 planos de la imagen YUV para transformarlos en RGB usando:
# https://stackoverflow.com/questions/60729170/python-opencv-converting-planar-yuv-420-image-to-rgb-yuv-array-format
def YUVtoRGB(yp, up, vp):
    print("la imagen llegó como lista, pero nada mas")

    # Building the input:
    ###############################################################################
    img = cv2.imread("ejemplo.jpg")

    # yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # y, u, v = cv2.split(yuv)

    # Convert BGR to YCrCb (YCrCb apply YCrCb JPEG (or YCC), "full range",
    # where Y range is [0, 255], and U, V range is [0, 255] (this is the default JPEG format color space format).
    yvu = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, v, u = cv2.split(yvu)
    print(u)
    print(u.length())

    # convierte las list en numpy arrays:
    # y = np.array(yp)
    upp = np.array(up)
    print(upp.length())
    # v = np.array(vp)

    # Downsample U and V (apply 420 format).
    u = cv2.resize(u, (u.shape[1] // 2, u.shape[0] // 2))
    v = cv2.resize(v, (v.shape[1] // 2, v.shape[0] // 2))

    # Open In-memory bytes streams (instead of using fifo)
    f = io.BytesIO()

    # Write Y, U and V to the "streams".
    f.write(y.tobytes())
    f.write(u.tobytes())
    f.write(v.tobytes())

    f.seek(0)
    ###############################################################################

    # Read YUV420 (I420 planar format) and convert to BGR
    ###############################################################################
    data = f.read(
        y.size * 3 // 2
    )  # Read one frame (number of bytes is width*height*1.5).

    # Reshape data to numpy array with height*1.5 rows
    yuv_data = np.frombuffer(data, np.uint8).reshape(y.shape[0] * 3 // 2, y.shape[1])

    # Convert YUV to BGR
    bgr = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420)

    return bgr


# usa un métdo de chatgpt para convertir las 3 listas en un rgb.
# este método se concentra en la creación de los numpy array
def YUVtoRGB2(height, width, yp, up, vp):
    # print("la imagen llegó como lista, pero nada mas")
    # se agrega un elemento más a la lista para que entre en el rango
    up.append(0)

    # crea 3 matrices con ceros con el tamaño y dimensiones adecuadas para cada plano
    y_matrix = np.zeros((height, width), dtype=np.uint8)
    u_matrix = np.zeros((height // 2, width // 2), dtype=np.uint8)
    v_matrix = np.zeros((height // 2, width // 2), dtype=np.uint8)

    # llenar las matrices con los datos de las listas
    # llenado de y_matrix
    cont = 0
    for i in range(0, height):
        for j in range(0, width):
            y_matrix[i, j] = yp[cont]
            cont += 1

    # llenado de u_matrix y v_matrix
    cont = 0
    for i in range(0, (height // 2)):
        for j in range(0, (width // 2)):
            u_matrix[i, j] = up[cont]
            v_matrix[i, j] = up[cont + 1]
            cont += 2

    # Crear el numpy array de YUV420
    yuv420_frame = np.zeros((height + height // 2, width), dtype=np.uint8)
    yuv420_frame[:height, :] = y_matrix
    yuv420_frame[height:, : width // 2] = u_matrix
    yuv420_frame[height:, width // 2 :] = v_matrix

    # se muestra la imagen
    cv2.imshow("Imagen yuv", yuv420_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # convertir el yuv frame en un rgb frame
    rgb_frame = cv2.cvtColor(yuv420_frame, cv2.COLOR_YUV2RGB_I420)

    # procesos extra de stack overlfow
    # muestra el
    # rgb_frame = cv2.cvtColor(yuv420_frame, cv2.COLOR_YUV2BGR_I420)
    return rgb_frame


# function to handle incoming connections from the client.
# This function is called whenever a client connects to the server.
@socketio.on("connect")
def test_connect():
    print("Connected")
    emit("my response", {"data": "Connected"})


# function to handle incoming images from the client.
# This function is called whenever an “image” event is received from the client.
@socketio.on("image")
def receive_image(height, width, yp, up, vp):
    print("Recibí una imagen muy bonita\n")
    # print("Received lists1: ", yp)
    # print("Received lists2: ", up)
    # print("Received lists3: ", vp)

    # Decode the base64-encoded image data (bueno)
    # image = base64_to_image(base64_image)

    # recibe una imagen como una lista de bytes y lo convierte en una imagen
    # image = process_camera_image(base64_image)

    # np_array = procesar_imagen(RGBA_image)

    # de una vez convierte la List<List<int>> en un RGB
    # image = YUVtoRGB(yp, up, vp)

    # intenta pasar la imagen yuv a rgb pero con un método sacado de chatgpt
    image_rgb = YUVtoRGB2(height[0], width[0], yp, up, vp)

    print("iniciando proyecto de malla facial")

    # soluciones que mandamos a llamar desde mediapipe para la malla y para dibujar
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    # definimos las características del detector de rostros
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
    ) as face_mesh:
        # se define la imagen de entrada
        # image = cv2.imread("lib\\python\\rostros\\rostro1.jpeg")
        # sacamos las dimensiones necesarias de la imagen
        # height, width, _ = image.shape
        # traformar la image en RGB, porque así la puede leer mediapipe
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image_rgb = cv2.cvtColor(np_array, cv2.COLOR_RGBA2RGB)

        # aplicamos la detección con face mesh
        results = face_mesh.process(image_rgb)
        # imprimimos los resultados como un json
        # print("Face landmarks: \n", results.multi_face_landmarks)

        # para dibujar los puntos encima de la imagen (si es que detectó un rostro)
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                # face_landmarks son los puntos, mp_face_mesh.FACEMESH_CONTOURS remarca zonas clave como labios y ojos
                mp_drawing.draw_landmarks(
                    image_rgb,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    # características del punto y la línea
                    mp_drawing.DrawingSpec(
                        color=(
                            0,
                            255,
                            255,
                        ),
                        thickness=1,
                        circle_radius=1,
                    ),
                    mp_drawing.DrawingSpec(
                        # _,
                        thickness=1,
                    ),
                )
    # se muestra la imagen
    cv2.imshow("Image", image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(str(results.multi_face_landmarks))
    # Send the processed image back to the client
    emit("processed_image", str(results.multi_face_landmarks))


# rutas para el servidor web
# ruta inicial del servidor
@app.route("/")
# quiero que la ruta reciba una imagen en el formato qe le gusta a mediapipe
def index():
    print("entramos al index ya con una imagen recibida")
    # lo que yo quiero retornar en reaildad es un json o  string de coordenadas
    return render_template("index.html")


# correr el servidor
if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000, host="0.0.0.0")

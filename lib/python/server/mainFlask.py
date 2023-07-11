# importaciones
# pip install flask-socketio==5.2.0
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, send, emit

# importación de librerías
import cv2
import mediapipe as mp
import base64

# inicializaoms le servidor web
app = Flask(__name__, static_folder="./templates/static")
# inicializar la conexión con websockets
app.config["SECRET_KEY"] = "secret"
socketio = SocketIO(app)


# helper function that will convert a base64-encoded image to a NumPy array that can be processed using OpenCV.
def base64_to_image(base64_string):
    # falta perfeccionar esta parte
    imgdata = base64.b64decode(base64_string)
    filename = "some_image.jpg"
    with open(filename, "wb") as f:
        f.write(imgdata)

    # se define la imagen de entrada
    imageCV2 = cv2.imread("lib\\python\\server\\some_image.jpg")
    return imageCV2


# function to handle incoming connections from the client.
# This function is called whenever a client connects to the server.
@socketio.on("connect")
def test_connect():
    print("Connected")
    emit("my response", {"data": "Connected"})


# function to handle incoming images from the client.
# This function is called whenever an “image” event is received from the client.
@socketio.on("image")
def receive_image(image):
    print("Recibí una imagen muy bonita")

    # Decode the base64-encoded image data
    image = base64_to_image(image)

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
        image = cv2.imread("lib\\python\\rostros\\rostro1.jpeg")
        # sacamos las dimensiones necesarias de la imagen
        height, width, _ = image.shape
        # traformar la image en RGB, porque así la puede leer mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # aplicamos la detección con face mesh
        results = face_mesh.process(image_rgb)
        # imprimimos los resultados como un json
        # print("Face landmarks: \n", results.multi_face_landmarks)

        # para dibujar los puntos encima de la imagen (si es que detectó un rostro)
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                # face_landmarks son los puntos, mp_face_mesh.FACEMESH_CONTOURS remarca zonas clave como labios y ojos
                mp_drawing.draw_landmarks(
                    image,
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
                        _,
                        thickness=1,
                    ),
                )

    # Send the processed image back to the client
    emit("processed_image", str(results.multi_face_landmarks))


# rutas para el servidor web
# ruta inicial del servidor
@app.route("/")
# quiero que la ruta reciba una imagen en el formato qe le gusta a mediapipe
def index():
    print("entramos al index ya con una imagen recibida")
    # lo que yo quiero retornar en reaildad es un json o  string de coordenadas
    # return "hola mundo"
    return render_template("index.html")


# correr el servidor
if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000, host="0.0.0.0")

# importación de librerías
import cv2
import mediapipe as mp


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
    print("Face landmarks: \n", results.multi_face_landmarks)

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

    # se muestra la imagen
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# importación de librerías
import cv2
import mediapipe as mp


print("iniciando proyecto de malla facial")

# soluciones que mandamos a llamar desde mediapipe para la malla y para dibujar
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# indicamos que va a usar la cámara para capturar video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


# definimos las características del detector de rostros
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:
    # se define que el video va a ser el de la misma cámara
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)  # para que se vea como espejo
        # apicamos RGB para que mediapipe lo pueda leer
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        # aplicamos la detección con face mesh
        results = face_mesh.process(frame_rgba)

        # para dibujar los puntos encima del video (si es que se detectó un rostro)
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                # face_landmarks son los puntos, mp_face_mesh.FACEMESH_CONTOURS remarca zonas clave como labios y ojos
                mp_drawing.draw_landmarks(
                    frame,
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
                        thickness=1,
                    ),
                )

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

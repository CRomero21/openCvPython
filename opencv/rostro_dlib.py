import cv2
import dlib
import numpy as np
import time

# Función para calcular la relación de aspecto del ojo (EAR)
def calculate_ear(eye_points):
    # Calcula las distancias euclidianas entre los puntos verticales
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    # Calcula la distancia euclidiana entre los puntos horizontales
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    # EAR = (A + B) / (2.0 * C)
    ear = (A + B) / (2.0 * C)
    return ear
# Inicializar dlib para la detección de rostros y puntos clave faciales
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Descargar el modelo

# Índices de los puntos clave para los ojos
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))

# Umbral para determinar si los ojos están cerrados
EAR_THRESHOLD = 0.2
CLOSED_DURATION_THRESHOLD = 1  # En segundos

# Variables para seguimiento
eyes_closed_start_time = None  # Tiempo inicial de ojos cerrados
eyes_closed_printed = False  # Verifica si ya se imprimió el mensaje

# Captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Reducir la resolución de la cámara
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Ancho de la imagen
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Altura de la imagen

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    eyes_are_closed = False

    for face in faces:
        # Detectar puntos clave faciales
        landmarks = predictor(gray, face)
        landmarks_points = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Extraer puntos de los ojos
        left_eye = landmarks_points[LEFT_EYE_POINTS]
        right_eye = landmarks_points[RIGHT_EYE_POINTS]

        # Calcular EAR para ambos ojos
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Determinar si los ojos están cerrados
        if ear < EAR_THRESHOLD:
            eyes_are_closed = True
            if eyes_closed_start_time is None:  # Registrar el inicio
                eyes_closed_start_time = time.time()
        else:
            eyes_are_closed = False
            eyes_closed_start_time = None
            eyes_closed_printed = False  # Reiniciar impresión

        # Dibujar los ojos en la imagen
        for point in left_eye:
            cv2.circle(frame, tuple(point), 2, (255, 0, 0), -1)
        for point in right_eye:
            cv2.circle(frame, tuple(point), 2, (255, 0, 0), -1)

    # Verificar si los ojos han estado cerrados el tiempo suficiente
    if eyes_are_closed and eyes_closed_start_time:
        elapsed_time = time.time() - eyes_closed_start_time
        if elapsed_time >= CLOSED_DURATION_THRESHOLD and not eyes_closed_printed:
            print("Ojos cerrados")
            eyes_closed_printed = True
        cv2.putText(frame, "Ojos Cerrados", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Ojos Abiertos", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Eye State Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np

def detect_ojos(img):
    """Detecta los ojos en una imagen y determina si están abiertos o cerrados.

    Args:
        img: La imagen de entrada.

    Returns:
        La imagen con los rectángulos alrededor de los ojos y el texto indicando si están abiertos o cerrados.
    """

    # Cargar clasificadores
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    left_eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    right_eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        # Detectar ojos izquierdo y derecho
        left_eyes = left_eye_cascade.detectMultiScale(roi_gray)
        right_eyes = right_eye_cascade.detectMultiScale(roi_gray)

        # Procesar cada ojo
        try:
            for (ex, ey, ew, eh) in left_eyes + right_eyes:
                cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

                # Región de interés del ojo
                roi_eye = gray[y+ey:y+ey+eh, x+ex:x+ex+ew]

                # ... (resto de tu código para procesar el ojo)
                # Por ejemplo, calcular características, clasificar, etc.

        except TypeError:
            # Si ocurre un error de tipo, significa que una lista está vacía
            if len(left_eyes) > 0:
                for (ex, ey, ew, eh) in left_eyes:
                    # ... (tu código para procesar el ojo izquierdo)
            elif len(right_eyes) > 0:
                for (ex, ey, ew, eh) in right_eyes:
                    # ... (tu código para procesar el ojo derecho)

    return img

# Capturar video de la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = detect_ojos(frame)
    cv2.imshow('Detección de ojos', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
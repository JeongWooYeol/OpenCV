import cv2, numpy as np


face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./data/haarcascade_eye.xml')

def distortion(rows, cols, type = 0):
    map_y, map_x = np.indices((rows, cols), dtype = np.float32)

    map_lenz_x = (2 * map_x - cols) / cols
    map_lenz_y = (2 * map_y - rows) / rows

    r, theta = cv2.cartToPolar(map_lenz_x, map_lenz_y)

    if type == 0:
        r[r < 1] = r[r < 1] ** 3 # convex lens
    else:
        r[r < 1] = r[r < 1] ** 0.5 # concave lens

    mapx, mapy = cv2.polarToCart(r, theta)
    mapx = ((mapx + 1) * cols) / 2
    mapy = ((mapy + 1) * rows) / 2
    return (mapx, mapy)


def findFaces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    face_coords = []
    for (x, y, w, h) in faces:
        face_coords.append((x, y, w, h))
    return face_coords

def findEyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    eyes_coords = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex, ey, ew, eh) in eyes:
            eyes_coords.append((ex + x, ey + y, ew, eh))
    return eyes_coords



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

while True:
    ret, frame = cap.read()
    img1 = frame.copy()
    img2 = frame.copy()

    faces = findFaces(frame) # face detect
    for face in faces:
        x, y, w, h = face
        mapx, mapy = distortion(w, h, 1) # distortion
        roi = img1[y : y + h, x : x + w]
        convex = cv2.remap(roi, mapx, mapy, cv2.INTER_LINEAR)
        img1[y : y + h, x : x + w] = convex

    eyes = findEyes(frame) # eyes detect
    for eye in eyes:
        x, y, w, h = eye
        mapx, mapy = distortion(w, h)
        roi = img2[y : y + h, x : x + w]
        convex = cv2.remap(roi, mapx, mapy, cv2.INTER_LINEAR)
        img2[y : y + h, x : x + w] = convex

    merged = np.hstack((frame, img1, img2))
    cv2.imshow('Face Distortion', merged)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()



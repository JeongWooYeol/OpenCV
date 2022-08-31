import cv2

rate = 15 # mosaic rate
face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml') # create cascade classifier

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5, minSize = (50, 50))

    for (x, y, w, h) in faces:
        x = x - 5
        y = y - 15
        w = w + 10
        h = h + 30
        roi = frame[y : y + h, x : x + w]
        roi = cv2.resize(roi, (w // rate, h // rate)) # mosaic

        roi = cv2.resize(roi, (w, h), interpolation = cv2.INTER_AREA)
        frame[y : y + h, x : x + w] = roi
    cv2.imshow("result", frame)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()


import cv2

# mask image
mask = cv2.imread('../img/mask_hannibal.png')
h_mask, w_mask = mask.shape[:2]


face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml') # create cascade classifier
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rect = face_cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)

    for (x, y, w ,h) in face_rect:
        if h > 0 and w > 0:
            x = int(x + 0.1 * x)
            y = int(y + 0.4 * y)
            w = int(0.8 * w)
            h = int(0.8 * h)

            roi = frame[y : y + h, x : x + w]
            mask_resize = cv2.resize(mask, (w, h), interpolation = cv2.INTER_AREA)
            gray_mask = cv2.cvtColor(mask_resize, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray_mask, 50, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            masked_face = cv2.bitwise_and(mask_resize, mask_resize, mask = mask)
            masked_frame = cv2.bitwise_and(roi, roi, mask = mask_inv)
            frame[y : y + h, x : x + w] = cv2.add(masked_face, masked_frame)

    cv2.imshow("result", frame)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
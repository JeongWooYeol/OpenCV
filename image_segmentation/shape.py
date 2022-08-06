import cv2
import numpy as np

dir = '../img/5shapes.jpg'

img = cv2.imread(dir)
result = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    verticle = len(approx)

    if verticle == 3:
        shape = "triangle"
        color = (0, 0, 0)

    elif verticle == 4:
        x, y, w, h = cv2.boundingRect(contour)
        if w == h:
            shape = "Square"
            color = (125, 125, 0)
        else:
            shape = "rectangle"
            color = (0, 0, 255)

    elif verticle == 10:
        shape = "star"
        color = (0, 255, 0)

    elif verticle > 15:
        shape = "circle"
        color = (255, 0, 0)

    cv2.drawContours(result, [contour], -1, color, -1)
    cv2.putText(result, shape, (contour[0][0]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (100, 100, 100), 1)

cv2.imshow('img', img)
cv2.imshow('result', result)
cv2.waitKey()
cv2.destroyAllWindows()
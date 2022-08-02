import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    gray_frame = cv2.bilateralFilter(frame, 5, 75, 75)
    gray_edge = cv2.Canny(gray_frame, 70, 255)
    gray_edge[:, :] = 255 - gray_edge[:, :]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray_edge = cv2.erode(gray_edge, kernel)
    gray_edge = cv2.medianBlur(gray_edge, 5)
    gray_edge = cv2.cvtColor(gray_edge, cv2.COLOR_GRAY2BGR)

    color_frame = cv2.blur(frame, (10, 10))
    color_edge = cv2.bitwise_and(color_frame, gray_edge)

    cv2.imshow("result", np.hstack((gray_edge, color_edge)))

    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break

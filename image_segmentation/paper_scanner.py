import numpy as np
import cv2

dir = 'img_dir'
img = cv2.imread(dir)
draw = img.copy()
win_name = 'paper'
cv2.imshow(win_name, img)
cv2.waitKey()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow(win_name, img_gray)
cv2.waitKey()

th = cv2.GaussianBlur(img_gray, (3, 3), 0)
cv2.imshow(win_name, th)
cv2.waitKey()

edges = cv2.Canny(th, 75, 200)
cv2.imshow(win_name, edges)
cv2.waitKey()

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(draw, contours, -1, (0, 255, 0))
cv2.imshow(win_name, draw)
cv2.waitKey()

contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
for c in contours:
    verticles = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
    print(len(verticles))
    if len(verticles) == 4:
        print(verticles)
        break


x_list = []
y_list = []
for i in verticles:
    for x, y in i:
        cv2.circle(draw, (x, y), 5, (0, 255, 0), -1)
        x_list.append(x)
        y_list.append(y)
cv2.imshow(win_name, draw)
cv2.waitKey()

verticles = verticles.reshape(4, 2)
sm = verticles.sum(axis = 1)
diff = np.diff(verticles, axis = 1)

TL = verticles[np.argmin(sm)]
TR = verticles[np.argmin(diff)]
BR = verticles[np.argmax(sm)]
BL = verticles[np.argmax(diff)]

pts1 = np.float32([TL, TR, BR, BL])

width = max(abs(BR[0] - BL[0]), abs(TR[0] - TL[0]))
height = max(abs(BR[1] - TR[1]), abs(BL[1] - TL[1]))

pts2 = np.float32([[0,0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
mtrx = cv2.getPerspectiveTransform(pts1, pts2)

result = cv2.warpPerspective(img, mtrx, (width, height))
cv2.imshow(win_name, result)
cv2.waitKey()

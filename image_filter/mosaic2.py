import numpy as np
import cv2

dir = 'img_dir'
win_title = "mosaic2"
img = cv2.imread(dir)

while True:
    x, y, w, h = cv2.selectROI(win_title, img, False)
    if w > 0 and h > 0:
        img[y : y + h, x : x + w, : ] = cv2.blur(img[y : y + h, x : x + w, : ] , (10, 10))
        cv2.imshow(win_title, img)
    else:
        break

cv2.destroyAllWindows()
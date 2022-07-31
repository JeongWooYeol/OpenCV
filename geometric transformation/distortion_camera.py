import numpy as np
import cv2

rows, cols = 480, 320
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cols)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rows)
_, frame = cap.read()

exp1 = 0.5
exp2 = 2.0
scale = 1

# concave
mapy, mapx = np.indices((rows, cols), dtype = np.float32)
mapx = 2 * mapx / (cols - 1) - 1
mapy = 2 * mapy / (rows - 1) - 1
r, theta = cv2.cartToPolar(mapx, mapy)

r[r<scale] = r[r<scale] ** exp1
mapx, mapy = cv2.polarToCart(r, theta)
mapx = ((mapx + 1) * cols - 1) / 2
mapy = ((mapy + 1) * rows - 1) / 2

# convex
mapy, mapx = np.indices((rows, cols), dtype = np.float32)
mapx = 2 * mapx / (cols - 1) - 1
mapy = 2 * mapy / (rows - 1) - 1
r, theta = cv2.cartToPolar(mapx, mapy)

r[r<scale] = r[r<scale] ** exp2
mapx, mapy = cv2.polarToCart(r, theta)
mapx = ((mapx + 1) * cols - 1) / 2
mapy = ((mapy + 1) * rows - 1) / 2

# sin, cos
mapy, mapx = np.indices((rows, cols), dtype = np.float32)
sinx = mapx + 15 * np.sin(mapy / 20)
cosy = mapy + 15 * np.cos(mapx / 20)

# mirror
maphy, maphx = np.indices((rows, cols), dtype = np.float32)
mapvy, mapvx = np.indices((rows, cols), dtype = np.float32)
maphx[ : , cols // 2 : ] = cols - maphx[ : , cols // 2 : ] - 1
mapvy[rows // 2 : , : ] = rows - mapvy[rows // 2 : , : ] - 1

while True:
    _, frame = cap.read()
    distorted_concave = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    distorted_convex = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    distorted_sin_cos = cv2.remap(frame, sinx, cosy, cv2.INTER_LINEAR)
    distorted_hmirror = cv2.remap(frame, maphx, maphy, cv2.INTER_LINEAR)
    distorted_vmirror = cv2.remap(frame, mapvx, mapvy, cv2.INTER_LINEAR)
    frame = cv2.resize(frame, (320, 480))
    r1 = np.hstack([frame, distorted_concave, distorted_convex])
    r2 = np.hstack([distorted_sin_cos, distorted_vmirror, distorted_hmirror])
    result = np.vstack([r1, r2])
    cv2.imshow("distorted", result)
    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

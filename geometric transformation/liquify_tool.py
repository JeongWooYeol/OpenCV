import cv2
import numpy as np

def liquify(img, cx1, cy1, cx2, cy2):
    x, y, w, h = cx1 - size, cy1 - size, size * 2, size * 2
    roi = img[y : y + h, x : + x + w].copy()
    result = roi.copy()

    box1 = [[[0, 0], [w, 0], [size, size]],
            [[0, 0], [0, h], [size, size]],
            [[0, h], [size, size], [w, h]],
            [[w, 0], [size, size], [w, h]]]
    box2 = [[[0, 0], [w, 0], [cx2 - x, cy2 - y]],
            [[0, 0], [0, h], [cx2 - x, cy2 - y]],
            [[0, h], [cx2 - x, cy2 - y], [w, h]],
            [[w, 0], [cx2 - x, cy2 - y], [w, h]]]

    for i in range(4):
        matrix = cv2.getAffineTransform(np.float32(box1[i]), np.float32(box2[i]))
        warped = cv2.warpAffine(roi.copy(), matrix, (w, h), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REFLECT_101)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(box2[i]), (255, 255, 255))

        warped = cv2.bitwise_and(warped, warped, mask = mask)
        result = cv2.bitwise_and(result, result, mask = cv2.bitwise_not(mask))
        result = result + warped

    img[y : y + h, x : x + w] = result
    return img

def onMouse(event, x, y, flags, param):
    global isDragging
    global img, mx, my
    if event == cv2.EVENT_MOUSEMOVE:
        if isDragging:
            img_draw = img.copy()
            cv2.rectangle(img_draw, (x - size, y - size), (x + size , y + size), (255, 0, 0))
            cv2.imshow(win_title, img_draw)

    elif event == cv2.EVENT_LBUTTONDOWN:
        isDragging = True
        mx, my = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        if isDragging:
            isDragging = False
            img = liquify(img, mx, my, x, y)
            cv2.imshow(win_title, img)


if __name__ == '__main__':
    win_title = 'liquify'
    dir = 'img_directory'
    img = cv2.imread(dir)
    h, w = img.shape[:2]
    size = 50
    isDragging = False
    cv2.imshow(win_title, img)
    cv2.setMouseCallback(win_title, onMouse)
    while True:
        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            break

    cv2.destroyAllWindows()



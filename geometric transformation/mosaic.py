import cv2

win_title = "mosaic"
dir = 'directory'
img = cv2.imread(dir)
rate = 15

while True:
    x, y, w, h = cv2.selectROI(win_title, img, False)
    if w and h:
        roi = img[y : y + h, y : y + w]
        roi = cv2.resize(roi, (w // rate, h // rate))
        roi = cv2.resize(roi, (w, h), cv2.INTER_AREA)
        img[y : y + h, x : x + w] = roi
        cv2.imshow(win_title, img)
    else:
        break

cv2.destroyAllWindows()


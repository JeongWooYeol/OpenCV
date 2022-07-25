# simple image alpha blending
import numpy as np
import cv2

dir1 = 'img1_directory'
dir2 = 'img2_directory'


blending_area = 15
img1 = cv2.imread(dir1)
img2 = cv2.imread(dir2)
result = np.zeros_like(img1)

height, width = img1.shape[:2]
middle = width // 2

result[:, :middle, :] = img1[:, :middle, :]
result[:, middle:, :] = img2[:, middle:, :]
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('half', result)
cv2.waitKey()
start = middle - (blending_area * width // 100) // 2

for i in range((middle - start) * 2):
    beta = (i) / 100
    alpha = 1 - beta
    result[:, start + i] = img1[:, start + i] * alpha + \
                                              img2[:, start + i] * beta


cv2.imshow("result", result)
cv2.waitKey()
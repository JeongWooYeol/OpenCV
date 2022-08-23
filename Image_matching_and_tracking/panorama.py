import cv2, numpy as np


imgL_path = '../img/restaurant1.jpg'
imgR_path = '../img/restaurant2.jpg'
imgL = cv2.imread(imgL_path)
imgR = cv2.imread(imgR_path)

hl, wl = imgL.shape[:2]
hr, wr = imgR.shape[:2]
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# descriptor 정의 및 keypoint, feature 계산
descriptor = cv2.xfeatures2d.SIFT_create()
(keypointL, featureL) = descriptor.detectAndCompute(imgL, None)
(keypointR, featureR) = descriptor.detectAndCompute(imgR, None)

# BFmatcher 생성 & knnMatch 사용
matcher = cv2.DescriptorMatcher_create("BruteForce")
matches = matcher.knnMatch(featureR, featureL, 2)

# good match 찾기
good_matches = []
for match in matches:
    if len(match) == 2 and match[0].distance < match[1].distance * 0.75:
        good_matches.append((match[0].trainIdx, match[0].queryIdx))

if len(good_matches) > 4:
    ptsL = np.float32([keypointL[i].pt for (i, _) in good_matches])
    ptsR = np.float32([keypointR[i].pt for (_, i) in good_matches])
    mtrx, status = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 4.0)
    # 원근 변환 행렬로 오른쪽 사진을 원근 변환, 결과 이미지 크기는 사진 2장 크기
    panorama = cv2.warpPerspective(imgR, mtrx, (wr + wl, hr))
    # 왼쪽 사진을 원근 변환한 왼쪽 영역에 합성
    panorama[0:hl, 0:wl] = imgL
else:
    panorama = imgL
cv2.imshow("Image Left", imgL)
cv2.imshow("Image Right", imgR)
cv2.imshow("Panorama", panorama)
cv2.waitKey(0)

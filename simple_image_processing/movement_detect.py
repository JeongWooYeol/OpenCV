# 움직임 감지

import cv2
import numpy as np

thresh = 15 # 달라진 픽셀 값 기준치
max_diff = 5 # 달라진 픽셀 갯수 기준치

cap = cv2.VideoCapture(0) # 카메라 세팅
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

if cap.isOpened():

    ret, a = cap.read()
    ret, b = cap.read()

    while ret:
        ret, c = cap.read()
        draw = c.copy()
        if not ret:
            break

        a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        c_gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

        diff1 = cv2.absdiff(a_gray, b_gray)
        diff2 = cv2.absdiff(b_gray, c_gray)

        _, diff1_t = cv2.threshold(diff1, thresh, 255, cv2.THRESH_BINARY)
        _, diff2_t = cv2.threshold(diff2, thresh, 255, cv2.THRESH_BINARY)
        # 차이가 15이상 나는 픽셀들은 255, 아니면 0으로 정리

        diff = cv2.bitwise_and(diff1_t, diff2_t)

        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, k)

        # 차이가 발생한 픽셀이 갯수 판단 후 사각형 그리기
        diff_cnt = cv2.countNonZero(diff)
        if diff_cnt > max_diff:
            nzero = np.nonzero(diff)  # 0이 아닌 픽셀의 좌표 얻기(y[...], x[...])
            cv2.rectangle(draw, (min(nzero[1]), min(nzero[0])), \
                          (max(nzero[1]), max(nzero[0])), (0, 255, 0), 2)
            cv2.putText(draw, "Motion Detect", (10, 30), \
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        # 컬러 영상과 스레시홀드 영상을 붙여 출력
        stacked = np.hstack((draw, cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)))
        cv2.imshow('motion sensor', stacked)

        # 영상 순서 변경
        a = b
        b = c

        if cv2.waitKey(1) & 0xFF == 27: # esc 누르면 종료
            break

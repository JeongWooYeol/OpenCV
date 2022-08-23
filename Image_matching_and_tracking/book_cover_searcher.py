import cv2, glob, numpy as np
# 100장의 책 표지가 존재합니다.
# 하얀색 사각형 안에 책을 두고 스페이스바를 눌러 촬영을 합니다.
# esc를 눌러 촬영한 책 표지로 검색 시작
# 해당 책 표지와 가장 유사한 책 표지를 보여주고 정확도도 보여줍니다.



# 검색 설정 변수
ratio = 0.7
MIN_MATCH = 10
book_path = "./book_path"

# ORB feature detector 생성
detector = cv2.ORB_create()

# Flann 매칭기 객체 생성
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)
search_params=dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)


# 책 표지 검색 함수
def search(img):
    gray_captureImg2= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoint1, desc1 = detector.detectAndCompute(gray_captureImg2, None)

    results = {}
    # 책 커버 보관 디렉토리 경로
    cover_paths = glob.glob(book_path)
    for cover_path in cover_paths:
        # 책 검색
        cover = cv2.imread(cover_path)
        cv2.imshow('Searching...', cover)
        cv2.waitKey(10)

        # 검색한 책 매칭
        gray2 = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
        keypoint2, desc2 = detector.detectAndCompute(gray2, None)
        matches = matcher.knnMatch(desc1, desc2, 2)

        # 좋은 매칭 선별
        good_matches = [m[0] for m in matches \
                        if len(m) == 2 and m[0].distance < m[1].distance * ratio]

        if len(good_matches) > MIN_MATCH:
            # 좋은 매칭점으로 원본과 대상 영상의 좌표 구하기
            src_pts = np.float32([keypoint1[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([keypoint2[m.trainIdx].pt for m in good_matches])

            # 원근 변환 행렬 구해 정상치 비율 계산
            mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            accuracy = float(mask.sum()) / mask.size
            results[cover_path] = accuracy
    cv2.destroyWindow('Searching...')
    if len(results) > 0:
        results = sorted([(v, k) for (k, v) in results.items() \
                          if v > 0], reverse=True)
    return results


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No frame!!")
        break
    else:
        h, w = frame.shape[:2]
        left = w // 3
        right = left * 2
        top = h // 5
        bottom = top * 4
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 4)
        cv2.imshow("camera", frame)
        key = cv2.waitKey(10)
        if key & 0xFF == 27:
            print("exit")
            break
        elif key == ord(" "):
            captureImg = frame[top : bottom, left : right]
            cv2.imshow("captureImg", captureImg)
cap.release()

if captureImg is not None:
    gray_captureImg = cv2.cvtColor(captureImg, cv2.COLOR_BGR2GRAY)
    result = search(captureImg)
    if len(result) <= 0:
        print("No matched book cover found...")
    else:
        for (i, (accuracy, cover_path)) in enumerate(result):
            print(i, cover_path, ":", accuracy)
            if i == 0:
                cover = cv2.imread(cover_path)
                cv2.putText(cover, ("Acuuracy: %.2f%%"%(accuracy * 100)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("result", cover)
cv2.waitKey()
cv2.destroyAllWindows()
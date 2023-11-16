import cv2
import numpy as np

# 카메라 캡처
cap = cv2.VideoCapture(0)

# ArUco 딕셔너리 생성
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Charuco 보드 생성
charuco_board = cv2.aruco.CharucoBoard_create((5, 7), 0.04, 0.02, aruco_dict)

# 카메라 행렬 및 왜곡 계수 (실제 값으로 업데이트 필요)
camera_matrix = np.array([[1.0, 0.0, 320], [0.0, 1.0, 240], [0.0, 0.0, 1.0]])
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # 마커 및 보드 검출
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)
    if len(corners) > 0:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, charuco_board
        )

        if charuco_corners is not None and charuco_ids is not None:
            # Charuco 보드의 위치 및 방향 추정
            _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, charuco_board, camera_matrix, dist_coeffs
            )

            # 보드의 중심 계산
            board_center = np.mean(charuco_corners, axis=0).squeeze()

            # 화면 중심을 기준으로 변환
            screen_center = np.array([frame.shape[1] // 2, frame.shape[0] // 2])
            board_center -= screen_center.astype(board_center.dtype)

            # 화면에 보드 중심 좌표 표시
            cv2.circle(frame, tuple(screen_center.astype(int)), 5, (0, 255, 0), -1)  # 화면 중심
            cv2.circle(frame, tuple(board_center.astype(int)), 5, (0, 0, 255), -1)    # 보드 중심

    # 영상 출력
    cv2.imshow('Charuco Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

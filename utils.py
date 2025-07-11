import math
import cv2
import streamlit as st

# 영상 소스 설정
def get_video_capture(input_mode, video_path=None):
    if input_mode == "Webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.warning("⚠️ 웹캠을 열 수 없습니다. 영상 파일 모드로 전환해 주세요.")
            return None
        return cap
    elif input_mode == "Video file from path" and video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("🚫 비디오 파일을 열 수 없습니다. 경로를 확인해 주세요.")
            return None
        return cap
    else:
        return None

# # 가상 펜스 내 포함 여부 확인 함수
# def is_inside_fence(box, fence):
#     x_min, y_min, x_max, y_max = box
#     fx1, fy1, fx2, fy2 = fence
#     center_x = (x_min + x_max) / 2
#     center_y = (y_min + y_max) / 2
#     return fx1 <= center_x <= fx2 and fy1 <= center_y <= fy2


# 사람 바운딩 박스와 fence가 겹치는지 확인
def is_overlapping_fence(person_box, fence_box):
    px1, py1, px2, py2 = person_box
    fx1, fy1, fx2, fy2 = fence_box

    # 두 박스가 전혀 겹치지 않을 경우 False
    if px2 < fx1 or px1 > fx2 or py2 < fy1 or py1 > fy2:
        return False
    return True  # 겹침



def calculate_fence_and_max(frame):
    """
    frame: np.ndarray, OpenCV로 읽은 BGR 이미지 프레임

    returns:
        x1, y1: 정사각형(가상펜스)의 왼쪽 상단 좌표
        x2, y2: 정사각형의 오른쪽 하단 좌표
        w, h   : 프레임의 폭(width), 높이(height)
    """
    h, w = frame.shape[:2]
    # 대각선 길이
    diag = math.hypot(w, h)
    # 대각선의 30% 크기
    square_size = diag * 0.25
    # 중앙점
    cx, cy = w / 2, h / 2
    half = square_size / 2

    # 좌표 계산
    x1 = int(cx - half)
    y1 = int(cy - half)
    x2 = int(cx + half)
    y2 = int(cy + half)

    # 가장자리 값 보정 (0 ~ w/h 범위)
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    return x1, y1, x2, y2, w, h



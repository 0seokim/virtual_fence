import streamlit as st
import cv2 , math
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from utils import *

# Streamlit 페이지 설정
st.set_page_config(layout="wide")
st.title("YOLO Virtual Fence Detection")

# # 모델 불러오기 (최초 1회만 실행됨)
# @st.cache_resource
# def load_model():
#     model = YOLO("yolov8n.pt")  # 필요한 경우 custom 모델 경로로 변경
#     return model

# model = load_model()
@st.cache_resource
def load_models():
    return [
        YOLO(r"/mnt/c/Users/kimyo/OneDrive/바탕 화면/streamlit/model5_best.pt"),  # pretrained on data1
        YOLO(r"/mnt/c/Users/kimyo/OneDrive/바탕 화면/streamlit/model6_best.pt"),  # pretrained on data2
        YOLO(r"/mnt/c/Users/kimyo/OneDrive/바탕 화면/streamlit/model4_best.pt"),  # pretrained on data3
    ]

models = load_models()

# 입력 선택
input_mode = st.sidebar.selectbox("Select Input Mode", ["Webcam", "Video file from path"])
# 펜스 모양 선택
shape = st.sidebar.selectbox("Fence Shape", ["Rectangle", "Triangle", "Circle"])  # 🔧
video_path = None  # 기본값

if input_mode == "Video file from path":
    video_path = st.sidebar.text_input(
        "Enter full path to video file",
        value="/mnt/c/Users/kimyo/OneDrive/바탕 화면/streamlit/3562070-hd_1920_1080_30fps.mp4",  # 예시 경로
        # help="예: /home/user/videos/sample.mp4 또는 C:\\Users\\user\\Videos\\test.mp4"
    )

# 스트림릿 이미지 출력을 위한 공간
frame_display = st.empty()

# 해상도 계산
cap0 = get_video_capture(input_mode, video_path)
if not cap0:
    st.error("⚠️ 영상 소스를 열 수 없습니다.")
    st.stop()
ret, frame0 = cap0.read()
cap0.release()
if not ret:
    st.error("⚠️ 첫 프레임을 읽을 수 없습니다.")
    st.stop()

h, w = frame0.shape[:2]
diag = math.hypot(w, h)
square = int(diag * 0.3)
cx, cy = w//2, h//2
# 슬라이더 범위: 0 ~ w or h
st.sidebar.header("Virtual Fence Settings")
x1 = st.sidebar.slider("Fence X1", 0, w, max(0, cx-square//2), key="f_x1")
y1 = st.sidebar.slider("Fence Y1", 0, h, max(0, cy-square//2), key="f_y1")
x2 = st.sidebar.slider("Fence X2", 0, w, min(w, cx+square//2), key="f_x2")
y2 = st.sidebar.slider("Fence Y2", 0, h, min(h, cy+square//2), key="f_y2")


# 메인 처리 루프
def run_detection():

    # 세션 상태로부터 로컬 변수로 언팩
    f_x1 = st.session_state.f_x1
    f_y1 = st.session_state.f_y1
    f_x2 = st.session_state.f_x2
    f_y2 = st.session_state.f_y2

    cap = get_video_capture(input_mode, video_path)
    if not cap or not cap.isOpened():
        st.warning("비디오를 열 수 없습니다.")
        return

    fence_rect = (f_x1, f_y1, f_x2, f_y2)


    # first = True  # 첫 프레임 여부
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        

        all_boxes_norm, all_scores, all_classes = [], [], []
        h, w = frame.shape[:2]
        for model in models:
            results = model.predict(source=frame, classes=[0], verbose=False)[0]  # class 0 = person
            # boxes = [box.xyxy[0].cpu().numpy().tolist() for box in results.boxes]
            # scores = [float(box.conf[0]) for box in results.boxes]
            # classes = [int(box.cls[0]) for box in results.boxes]
            boxes_n, scores, classes = [], [], []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                boxes_n.append([x1/w, y1/h, x2/w, y2/h])
                scores.append(float(box.conf[0]))
                classes.append(int(box.cls[0]))
            all_boxes_norm.append(boxes_n)
            all_scores.append(scores)
            all_classes.append(classes)
            
        if any(all_boxes_norm):
            boxes_n, scores, classes = weighted_boxes_fusion(
             all_boxes_norm, all_scores, all_classes,
             iou_thr=0.5, skip_box_thr=0.25
            )
        else:
            boxes, scores, classes = [], [], []

        boxes = []
        boxes = [(int(bx[0]*w), int(bx[1]*h), int(bx[2]*w), int(bx[3]*h)) for bx in boxes_n]
        
        annotated_frame = frame.copy()
        person_inside = False

        for (x_min, y_min, x_max, y_max), score, cls in zip(boxes, scores, classes):
            # 클래스가 0(person)인지 재확인
            if cls == 0:
                inside = is_overlapping_fence((x_min, y_min, x_max, y_max), fence_rect)
                if inside:
                    person_inside = True

                color = (0, 0, 255) if inside else (255, 255, 0)
                label = f"Human {score:.2f}"
                cv2.rectangle(annotated_frame, (int(x_min), int(y_min)),
                              (int(x_max), int(y_max)), color, 2)
                cv2.putText(annotated_frame, label, (int(x_min), int(y_min)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # person
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                    box_coords = (x_min, y_min, x_max, y_max)
                    # fence_rect = (x1, y1, x2, y2)

                    inside = is_overlapping_fence(box_coords, fence_rect)
                    if inside:
                        person_inside = True

                    color = (0, 0, 255) if inside else (255, 255, 0)  # 빨간색 or 노란색
                    label = "Human (in fence)" if inside else "Human"
                    cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(annotated_frame, label, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        current_fence_color = (0, 0, 255) if person_inside else (0, 255, 0)

        # Fence 내부 채움 (연하게)
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (f_x1, f_y1), (f_x2, f_y2), current_fence_color, -1)
        alpha = 0.2
        annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)


        # 스트림릿 이미지 업데이트
        cv2.rectangle(annotated_frame, (f_x1, f_y1), (f_x2, f_y2), current_fence_color, 2)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_display.image(annotated_frame, channels="RGB")

    cap.release()

# 실행 버튼
if st.button("▶️ Start Detection"):
    run_detection()

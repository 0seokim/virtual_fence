import streamlit as st
import cv2 , math
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from utils import *
# Streamlit 페이지 설정
st.set_page_config(layout="wide")
st.title("YOLO Virtual Fence Detection")


model_mode = st.sidebar.selectbox(
    "Model Mode", ["Base Model", "Finetuned Models"]
)

# Single 모델 불러오기 
@st.cache_resource
def load_basemodel():
    model = YOLO("yolov8n.pt")  # 필요한 경우 custom 모델 경로로 변경
    return model

# Ensemble 모델들 로드
@st.cache_resource
def load_ensemble():
    return [
        YOLO(r"/mnt/c/Users/kimyo/OneDrive/바탕 화면/streamlit/model5_best.pt"),  # pretrained on data1
        YOLO(r"/mnt/c/Users/kimyo/OneDrive/바탕 화면/streamlit/model6_best.pt"),  # pretrained on data2
        YOLO(r"/mnt/c/Users/kimyo/OneDrive/바탕 화면/streamlit/model4_best.pt"),  # pretrained on data3
    ]

if model_mode == "Base Model":
    model = load_basemodel()            # 한 개
else:
    models = load_ensemble()


# 입력 선택
input_mode = st.sidebar.selectbox("Select Input Mode", ["Webcam", "Video file from path"])
# 펜스 모양 선택
shape = st.sidebar.selectbox("Fence Shape", ["Rectangle", "Triangle", "Circle"])  # 🔧

video_path = None  # 기본값

# 입력 비디오 경로 설정
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
    cap = get_video_capture(input_mode, video_path)
    if not cap or not cap.isOpened():
        st.warning("비디오를 열 수 없습니다.")
        return

    f_x1 = st.session_state.f_x1
    f_y1 = st.session_state.f_y1
    f_x2 = st.session_state.f_x2
    f_y2 = st.session_state.f_y2


    fence_rect = (f_x1, f_y1, f_x2, f_y2)

    # first = True  # 첫 프레임 여부
    while cap.isOpened():
        success, frame = cap.read()
        if not success:break
        
         # 🔧 모델 모드에 따라 predict 분기
        if model_mode == "Base Model":
            results = model.predict(source=frame, classes=[0], verbose=False)[0]
            boxes = [tuple(map(int, b.xyxy[0])) for b in results.boxes]
            scores = [float(b.conf[0]) for b in results.boxes]
            classes = [int(b.cls[0]) for b in results.boxes]
        else:
            # ensemble 로직 (models 리스트 사용)
            all_b, all_s, all_c = [], [], []
            h, w = frame.shape[:2]
            for m in models:
                results = m.predict(source=frame, classes=[0], verbose=False)[0]
                bn, sc, cl = [], [], []
                for b in results.boxes:
                    x1,y1,x2,y2 = b.xyxy[0]
                    bn.append([x1/w, y1/h, x2/w, y2/h])
                    sc.append(float(b.conf[0]))
                    cl.append(int(b.cls[0]))
                all_b.append(bn); all_s.append(sc); all_c.append(cl)
            if any(all_b):
                bn, scores, classes = weighted_boxes_fusion(all_b, all_s, all_c,
                                                           iou_thr=0.5,
                                                           skip_box_thr=0.25)
                boxes = [(int(x1*w), int(y1*h), int(x2*w), int(y2*h)) for x1,y1,x2,y2 in bn]
            else:
                boxes, scores, classes = [], [], []
        
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

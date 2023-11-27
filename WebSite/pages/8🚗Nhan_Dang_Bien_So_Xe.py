import streamlit as st
from ultralytics import YOLO
import cv2
import pytesseract
import time
from PIL import Image
import numpy as np

# Get all license plates in image
def get_plates(result, img):
    images = [] # Store all license plates
    boxes = result[0].boxes # List of all coordinates of license plates
    img = img.copy()
    for b in boxes:
        x1 = int(b.xyxy[0][0])
        y1 = int(b.xyxy[0][1])
        x2 = int(b.xyxy[0][2])
        y2 = int(b.xyxy[0][3])
        images.append(img[y1:y2, x1:x2].copy())
    return images

# OCR using Tesseract
def tesseract_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Perform image preprocessing if needed
    # ...
    pytesseract.pytesseract.tesseract_cmd = 'utility/B8_RecognitionLicensePlates/Tesseract-OCR/tesseract.exe'
    # Apply OCR using Tesseract
    config = '--oem 3 --psm 6'  # Tesseract OCR configuration
    text = pytesseract.image_to_string(gray, config=config)
    return text

# Get LP numbers
def get_LP_number(result, img):
    plates = get_plates(result, img)
    plate_numbers = [] # Store all LP numbers
    
    for plate in plates:
        number = tesseract_ocr(plate)
        plate_numbers.append(number)
    
    return plate_numbers

# Process single image
# Draw rectangle around plates and LP numbers
def draw_box(result, img):
    boxes = result[0].boxes # All coordinates of plates
    plate_numbers = get_LP_number(result, img) # All predicted LP numbers
    
    # For each LP coordinates and each predicted LP number of that LP
    for b, pnum in zip(boxes, plate_numbers): 
        x1 = int(b.xyxy[0][0])
        y1 = int(b.xyxy[0][1]) - 20 # Small adjust to make it look better
        x2 = int(b.xyxy[0][2])
        y2 = int(b.xyxy[0][3])
        # Draw rectangle around the LP
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Fill background of the predicted LP number
        cv2.rectangle(img, (x1, y1 + 22), ((x2), (y1)), (0, 255, 0), -1)
        text_xy = (x1 + 2, y1 + 18)  # Coordinate of predicted LP number
        # Add predicted LP number
        # img, text, position, font, font_scale, color, thickness
        cv2.putText(img, pnum, text_xy, 0, 0.7, 0, 2) 
    
    return img

# Process video 
def video_draw_box(vid_path, model, frame_placeholder):
    cap = cv2.VideoCapture(vid_path)
    # Process each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    
        result = model(frame)   # Predict position of LPs
        frame = draw_box(result, frame) # Draw rectangle and predicted LP numbers for current frame
        frame_placeholder.image(frame, channels="BGR")
        if cv2.waitKey(1) == ord('q'):
            break

        # Độ trễ giữa các khung hình (thay đổi giá trị nếu cần)
        time.sleep(1/100)  
def main():
    st.title("Nhận dạng biển số xe trực tiếp trên hình hoặc video")
    # Get weights and file path
    pre_trained_model = "utility/B8_RecognitionLicensePlates/best.pt"
    media_type = st.radio("Loại file tải lên: ", ["Ảnh", "Video"])
    file_path = st.file_uploader("Tải ảnh hoặc video lên", type=["jpg", "jpeg", "png", "mp4"])

    # Create model
    model = YOLO(pre_trained_model)

    if file_path is not None:
        if media_type == "Ảnh":
            file_bytes = np.asarray(bytearray(file_path.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.write("<h2 style='font-size: 24px;'> Ảnh gốc </h2>", unsafe_allow_html=True)
            st.image(img, channels="BGR")
            result = model(img)
            img = draw_box(result, img)
            st.write("<h2 style='font-size: 24px;'> Ảnh sau khi nhận diện </h2>", unsafe_allow_html=True)
            st.image(img, channels="BGR")
        
        elif media_type == "Video":
            st.write("<h2 style='font-size: 24px;'> Video gốc </h2>", unsafe_allow_html=True)
            st.video(file_path)
            st.write("<h2 style='font-size: 24px;'> Video sau khi nhận diện</h2>", unsafe_allow_html=True)
            frame_placeholder = st.empty()
            video_draw_box(file_path.name, model, frame_placeholder)
            
if __name__ == "__main__":
    main()

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import sys

# Add YOLOv5 directory to path (if needed)
sys.path.append('yolov5')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# YOLOv5 predefined classes
YOLOV5_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Preprocess query to extract meaningful words
def preprocess_query(query):
    words = word_tokenize(query.lower())
    stop_words = set(stopwords.words('english'))
    interrogative_terms = {'who', 'what', 'where', 'when', 'why', 'how', 'find'}
    meaningful_words = []
    for word, tag in pos_tag(words):
        if word not in stop_words and word not in interrogative_terms and (tag.startswith('N') or tag.startswith('V')):
            meaningful_words.append(word)
    return meaningful_words

def detect_objects_yolo(frame_path, model, device, meaningful_words):
    img = cv2.imread(frame_path)
    img0 = img.copy()
    img = letterbox(img, 640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=False)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    detected_meaningful_frames = []

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            labels = det[:, -1].cpu().numpy()
            label_names = [model.names[int(label)] for label in labels]

            found_words = set()
            for word in meaningful_words:
                for i, name in enumerate(label_names):
                    if word in name.lower():
                        xyxy = det[i, :4]
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        cv2.rectangle(img0, c1, c2, (255, 0, 0), 2)  # Draw bounding box
                        # Display object name above the bounding box
                        font_scale = 0.5
                        thickness = 1
                        color = (255, 255, 255)  # White color for text
                        cv2.putText(img0, name, (c1[0], c1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                        found_words.add(word)
            if len(found_words) == len(meaningful_words):
                # Save the frame with bounding boxes to a temporary file
                temp_frame_path = frame_path.replace(".jpg", "_detected.jpg")
                cv2.imwrite(temp_frame_path, img0)
                detected_meaningful_frames.append(temp_frame_path)

    return detected_meaningful_frames, img0

# Adjusted main function
def main(video_path, query):
    output_folder = os.path.join('uploads', 'output', query)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Extracting frames from {video_path}...")
    progress_step = max(total_frames // 10, 1)
    pbar = tqdm(total=total_frames)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
        pbar.update(1)
        pbar.set_description(f"Extracted: {count}/{total_frames}")

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()

    if count == 0:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend('yolov5s.pt', device=device)
    model.eval()

    meaningful_words = preprocess_query(query)
    print(f"Meaningful words extracted from query: {meaningful_words}")

    queried_frames_folder = os.path.join(output_folder, "_".join(meaningful_words))
    os.makedirs(queried_frames_folder, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(output_folder) if f.startswith('frame_') and f.endswith('.jpg')])
    detected_frames = []

    print(f"Detecting objects in frames using YOLOv5...")
    pbar = tqdm(total=len(frame_files))
    for frame_file in frame_files:
        frame_path = os.path.join(output_folder, frame_file)
        detected_frame_paths, processed_frame_img = detect_objects_yolo(frame_path, model, device, meaningful_words)
        if detected_frame_paths:  # Check if there are any detected frames
            for detected_frame_path in detected_frame_paths:
                processed_frame_path = os.path.join(queried_frames_folder, os.path.basename(detected_frame_path))
                cv2.imwrite(processed_frame_path, cv2.imread(detected_frame_path))
                detected_frames.append(processed_frame_path)

        pbar.update(len(detected_frame_paths))
        pbar.set_description(f"Frames processed: {len(detected_frames)}/{len(frame_files)}")

    pbar.close()

    # Inside the main function, after processing the frames:
    if len(detected_frames) > 0:
        frame = cv2.imread(detected_frames[0])
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Change to 'mp4v' if needed
        out = cv2.VideoWriter(os.path.join('uploads', f'processed_{query}.mp4'), fourcc, 30.0, (width, height))

        for frame_path in detected_frames:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            out.write(frame)

        out.release()
    else:
        return None

    return os.path.join('uploads', f'processed_{query}.mp4')
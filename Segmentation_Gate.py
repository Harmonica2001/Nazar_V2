import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

model = YOLO("yolo11s-seg.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

window_name = "YOLO Segmentation with Aspect Ratios"
save_dir = "Segmentation_Gate_photos"               # annotated full-frame saves
save_bb_dir = "Segmentation_Gate_photos_BB"         # bounding-box crops
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_bb_dir, exist_ok=True)

PERSON_CLASS_ID = 0
MIN_FLAG_DURATION = 0.5  # seconds

trackers = {}  # key -> (start_time or None, saved_bool)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    now = datetime.now().timestamp()

    results = model(frame, verbose=False)
    r = results[0]

    # Annotated image with boxes/masks/labels
    annotated = r.plot().copy()

    current_keys = set()

    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            if cls_id == PERSON_CLASS_ID:
                continue

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            # clamp coords to frame
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)

            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            aspect = w / h
            aspect_2f = h / w

            flag = 1 if ((aspect < 0.5 and aspect_2f > 2.5) or (aspect_2f < 0.5 and aspect > 2.5)) else 0

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            key = (cls_id, (cx // 20), (cy // 20))  # quantized key
            current_keys.add(key)

            if flag == 1:
                if key not in trackers or trackers[key][0] is None:
                    trackers[key] = (now, False)
                else:
                    start_time, saved = trackers[key]
                    if (not saved) and (now - start_time >= MIN_FLAG_DURATION):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"{timestamp}_cls{cls_id}_flag1.jpg"
                        filepath = os.path.join(save_dir, filename)
                        # save annotated full-frame
                        cv2.imwrite(filepath, annotated)

                        # save bounding-box crop (from original frame to preserve quality)
                        bb_crop = frame[y1:y2, x1:x2].copy()
                        if bb_crop.size != 0:
                            bb_filename = f"{timestamp}_cls{cls_id}_flag1_bb.jpg"
                            bb_path = os.path.join(save_bb_dir, bb_filename)
                            cv2.imwrite(bb_path, bb_crop)

                        trackers[key] = (start_time, True)
            else:
                trackers[key] = (None, False)

            # draw bold label on annotated
            cls_name = model.names[cls_id]
            label = f"{cls_name} w/h:{aspect:.2f} h/w:{aspect_2f:.2f} -> {flag}"
            text_pos = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 15)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            cv2.putText(annotated, label, text_pos, font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, label, text_pos, font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # cleanup trackers for absent keys
    for k in list(trackers.keys()):
        if k not in current_keys:
            del trackers[k]

    cv2.imshow("Raw View", frame)
    cv2.imshow("All Information", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

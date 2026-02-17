import cv2
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "Nazar_17.pt"
CAMERA_SOURCE = 0
CONF_THRESHOLD = 0.4
THINNESS_THRESHOLD = 0.5   # Adjust this
# ---------------------------------------

model = YOLO(MODEL_PATH)
model.overrides['verbose'] = False

cap = cv2.VideoCapture(CAMERA_SOURCE)

print("🔪 Knife Aspect Ratio Checker Started")
print("Press Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            conf = float(box.conf)
            cls = int(box.cls)
            class_name = model.names[cls].lower()

            if conf < CONF_THRESHOLD or class_name != "knife":
                continue

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            # 🔥 ORIENTATION-INDEPENDENT RATIO
            thinness_ratio = min(width, height) / max(width, height)

            valid = thinness_ratio < THINNESS_THRESHOLD
            color = (0, 255, 0) if valid else (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"Knife | Thin={thinness_ratio:.2f}"
            status = "VALID" if valid else "REJECTED"

            cv2.putText(
                frame,
                f"{label} [{status}]",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            print(
                f"Knife detected | Conf: {conf:.2f} | "
                f"Thinness Ratio: {thinness_ratio:.2f} | {status}"
            )

    cv2.imshow("Knife Aspect Ratio Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

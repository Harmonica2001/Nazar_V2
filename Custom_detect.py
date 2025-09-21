import cv2
from ultralytics import YOLO
import time
import os
from datetime import datetime


def gate_1(source, model, conf_threshold):
    global gate1_bool, item
    
    item = None
    cap = cv2.VideoCapture(source)
    
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection without printing logs
        results = model(frame, verbose=False)
        detected = 0  # default is 0

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf)
                class_id = int(box.cls)   # ✅ class index
                class_name = model.names[class_id]  # ✅ get class name (knife, alcohol, etc.)

                if confidence > conf_threshold:
                    detected = 1
                    print(f"Detected: {class_name} ({confidence:.2f})")  # ✅ print class type
                    gate1_bool = True
                    
                    # Example: check specifically for knife or alcohol
                    if class_name.lower() == "knife":
                        item = "knife"
                        print("Knife detected!")
                    elif class_name.lower() == "alcohol":
                        item = "alcohol"
                        print("Alcohol detected!")

                    return True   # ✅ Exit only when detection happens

        if detected == 0:
            print(0)

        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release only when done
    cap.release()
    cv2.destroyAllWindows()
    return False


def gate_2(source, model, conf_threshold):
    
    cap = cv2.VideoCapture(source)
    
    start_time = None  # timer for person detection

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        person_detected = False
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf)
                cls = int(box.cls)   # class index
                if cls == 0 and confidence > conf_threshold:  # person class is usually 0
                    person_detected = True
                    break

        if person_detected:
            if start_time is None:
                start_time = time.time()  # start timer
            elif time.time() - start_time >= 1:  # detected for 3 
                print("Person also detected")
                return True
        else:
            return False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False

def notif():
    global item
    
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    
    print(f"Human with {item} in possession")

    # Get current datetime
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Specify the paths for the folders
    folder1_path = r"C:\Users\ahmad\Nazar\instances"
    folder2_path = r"C:\Users\ahmad\Nazar\instance_logs"

    # Create the folders if they don't exist
    os.makedirs(folder1_path, exist_ok=True)
    os.makedirs(folder2_path, exist_ok=True)

    # Save the image
    image_path = os.path.join(folder1_path, f"danger_instance_{timestamp}.jpg")
    cv2.imwrite(image_path, frame)
    print(f"Image saved as '{image_path}'")

    # Save the log
    log_path = os.path.join(folder2_path, f"danger_instance_{timestamp}.txt")
    with open(log_path, 'w') as log_file:
        log_file.write(f"Detected human with {item} at {now}\n")
    print(f"Log saved as '{log_path}'")
        

model_2 = YOLO('yolo11n.pt')
# settings for Gate_1
model_1 = YOLO('Nazar_12.pt')
model_1.overrides['verbose'] = False  # disable internal logs
source_1 = 0
conf_threshold_1 = 0.6
# settings for Gate_2
model_2 = YOLO('yolo11n.pt')
model_2.overrides['verbose'] = False  # disable internal logs
source_2 = 0
conf_threshold_2 = 0.7

while True:
    gate1_bool = False

    while not gate1_bool:
        gate1_bool = gate_1(source_1, model_1, conf_threshold_1)  # ✅ update variable properly
        
    if gate1_bool and gate_2(source_2, model_2, conf_threshold_2):
        notif()
    else:
        print(f"{item} found, no Human")

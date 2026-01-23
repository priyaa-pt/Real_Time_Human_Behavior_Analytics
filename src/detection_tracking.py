import cv2
import os
import time
import math
import csv

# -------------------------------
# VIDEO PATH
# -------------------------------
VIDEO_PATH = os.path.join("data", "videos", "mall_cctv_01.mp4")

# -------------------------------
# PERSON DETECTOR (HOG)
# -------------------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# -------------------------------
# TRACKING STATE
# -------------------------------
tracked_objects = {}          # id -> centroid
last_detected_boxes = []      # boxes to draw every frame
next_object_id = 0

DISTANCE_THRESHOLD = 60       # for ID matching
PROCESS_EVERY_N_FRAMES = 3    # detection frequency


# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def get_centroid(x, y, w, h):
    return int(x + w / 2), int(y + h / 2)


def assign_id(centroid):
    global next_object_id

    for obj_id, prev_centroid in tracked_objects.items():
        if math.dist(centroid, prev_centroid) < DISTANCE_THRESHOLD:
            tracked_objects[obj_id] = centroid
            return obj_id

    tracked_objects[next_object_id] = centroid
    next_object_id += 1
    return next_object_id - 1


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():
    os.makedirs("logs", exist_ok=True)

    csv_file = open("logs/tracking_data.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["person_id", "x", "y", "timestamp"])

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
       fps = 25  # fallback FPS
    delay = int(1000 / fps)


    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        frame_count += 1

        # Resize for speed
        frame = cv2.resize(frame, (640, 360))



        # -------------------------------
        # DETECT PEOPLE (EVERY N FRAMES)
        # -------------------------------
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            boxes, _ = hog.detectMultiScale(
                gray,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.03
            )

            last_detected_boxes.clear()

            for (x, y, w, h) in boxes:
                centroid = get_centroid(x, y, w, h)
                obj_id = assign_id(centroid)
                last_detected_boxes.append((x, y, w, h, obj_id))

        # -------------------------------
        # DRAW BOXES (EVERY FRAME)
        # -------------------------------
        for (x, y, w, h, obj_id) in last_detected_boxes:
             csv_writer.writerow([obj_id, x, y, time.time()])
             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
             cv2.putText(
                frame,
                f"ID {obj_id}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.imshow("Real-Time Human Behavior Tracking", frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            print("Stopped by user.")
            break
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Program finished successfully.")


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    main()



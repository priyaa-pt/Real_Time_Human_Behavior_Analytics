import cv2
import os

# -----------------------------
# Video path (DO NOT hardcode absolute paths)
# -----------------------------
VIDEO_PATH = os.path.join(
    "data",
    "videos",
    "mall_cctv_01.mp4"
)

def main():
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found at {VIDEO_PATH}")

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError("Error opening video file")

    # Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video loaded successfully")
    print(f"FPS: {fps}")
    print(f"Resolution: {width} x {height}")
    print(f"Total frames: {total_frames}")

    # Frame reading loop
    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video reached")
            break

        # Display frame (verification only)
        cv2.imshow("Video Preview - Day 1", frame)

        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




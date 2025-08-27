import cv2
import threading
import time
import os
from pathlib import Path
from gender_age_utils import GenderPredicion  # Adjust this if your class is in a different file name
import queue

def main():
    video_src = "./test_data/face_det_test.mp4"
    # video_src = "rtsp://capgemini:cgsmartkcc@192.168.10.108:554/stream1"
    output_dir = "./output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Instantiate
    gender_detector = GenderPredicion()

    # Manually fix the queue import if missing
    gender_detector.camera_buffer = queue.Queue(maxsize=1)

    # Load models
    gender_detector.load_models()

    # Thread to feed frames into the buffer
    stop_event = threading.Event()
    feeder_thread = threading.Thread(target=gender_detector.fill_camera_buffer, args=(video_src, stop_event))
    feeder_thread.start()

    try:
        # Run inference loop (this will save outputs per frame)

        # Wait for at least one frame to be buffered (max 3s wait)
        max_wait_sec = 3
        waited = 0
        while gender_detector.camera_buffer.empty() and waited < max_wait_sec:
            print(f"Waiting for camera buffer... ({waited + 1}/{max_wait_sec})")
            time.sleep(1)
            waited += 1

        # Run inference loop
        gender_detector.run_gender_pred()

    except KeyboardInterrupt:
        print("Interrupted. Stopping threads.")
    finally:
        stop_event.set()
        feeder_thread.join()
        print("Processing complete.")

if __name__ == "__main__":
    main()

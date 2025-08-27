import cv2 
import math
import time
import argparse
from threading import Thread
import numpy as np
import queue


class GenderPredicion:

    def __init__(self):
        self.camera_buffer = queue.Queue(maxsize=1)
        self.faceProto = "./model_files/opencv_face_detector.pbtxt"
        self.faceModel = "./model_files/opencv_face_detector_uint8.pb"

        self.genderProto = "./model_files/gender_deploy.prototxt"
        self.genderModel = "./model_files/gender_net.caffemodel"
        self.ageProto = "./model_files/age_deploy.prototxt"
        self.ageModel = "./model_files/age_net.caffemodel"

        self.genderNet = None
        self.faceNet = None
        self.ageNet = None 

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.genderList = ['Male', 'Female']

        self.padding = 20
        self.frame_count = 0

    # ------------------------------------------------------
    # CAMERA CAPTURE
    # ------------------------------------------------------
    def fill_camera_buffer(self, video_src, stop_event=None):
        """Continuously reads frames from the video source and places them in a buffer."""
        if video_src.startswith("rtsp") or str(video_src).strip() == "0":
            print("The source is a live stream")
            cap = cv2.VideoCapture(video_src)
            if not cap.isOpened():
                print("Error: Unable to open RTSP stream")
                return

            while not stop_event.is_set():
                ret, frame = cap.read()
                # frame = cv2.imread("./test_data/couple1.jpg")
                if not ret:
                    print("End of RTSP stream or cannot read frame.")
                    if self.camera_buffer.full():
                        _ = self.camera_buffer.get(timeout=1)
                    self.camera_buffer.put(None, timeout=1)
                    break
                # print(f"----frame shape: {frame.shape}")
                # frame = cv2.flip(frame, 0) #flip the frame horizontally
                # frame = cv2.flip(frame, 1)
                if not self.camera_buffer.full():
                    self.camera_buffer.put(frame, timeout=1)
                else:
                    self.camera_buffer.get(timeout=1)  # Drop the oldest frame
                    self.camera_buffer.put(frame, timeout=1)

        if video_src.lower().endswith(".mp4") :

            print("The source is a recorded video")
            cap = cv2.VideoCapture(video_src)
            if not cap.isOpened():
                print("Error: Unable to open video")
                return

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("End of stream or cannot read frame.")
                    if self.camera_buffer.full():
                        _ = self.camera_buffer.get(timeout=1)
                    self.camera_buffer.put(None, timeout=1)
                    break
                # print(f"----frame shape: {frame.shape}")
                # frame = cv2.flip(frame, 0) #flip the frame horizontally
                # frame = cv2.flip(frame, 1)
                if not self.camera_buffer.full():
                    self.camera_buffer.put(frame, timeout=1)
                else:
                    self.camera_buffer.get(timeout=1)  # Drop the oldest frame
                    self.camera_buffer.put(frame, timeout=1)



    def load_models(self):
        self.genderNet = cv2.dnn.readNet(self.genderModel, self.genderProto)
        self.faceNet = cv2.dnn.readNet(self.faceModel, self.faceProto)
        self.ageNet = cv2.dnn.readNet(self.ageModel, self.ageProto)


        self.genderNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        self.faceNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        self.ageNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        
        print("GenderNet and FaceNet models loaded.")


    def getFaceBox(self, net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn, bboxes


    def run_gender_pred(self):
        print("Starting gender prediction...")

        while True:
            if not self.camera_buffer.empty():
                start_time = time.time()
                input_image = self.camera_buffer.get_nowait()

                if input_image is None:
                    print("Reached end of stream. Exiting gender prediction.")
                    return
                self.frame_count += 1
                frameFace, bboxes = self.getFaceBox(self.faceNet, input_image)
                if not bboxes:
                    print("No face Detected, Checking next frame")
                    continue

                for bbox in bboxes:
                    # print(bbox)
                    face = input_image[max(0,bbox[1]-self.padding):min(bbox[3]+self.padding,input_image.shape[0]-1),max(0,bbox[0]-self.padding):min(bbox[2]+self.padding, input_image.shape[1]-1)]

                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
                    self.genderNet.setInput(blob)
                    genderPreds = self.genderNet.forward()
                    gender = self.genderList[genderPreds[0].argmax()]
                    # print("Gender Output : {}".format(genderPreds))
                    print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

                    self.ageNet.setInput(blob)
                    agePreds = self.ageNet.forward()
                    age = self.ageList[agePreds[0].argmax()]
                    print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

                    label = "{},{}".format(gender, age)
                    cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                    cv2.imwrite(f"./output/test_output_{self.frame_count}.jpg", frameFace)

                end_time = time.time()
                print(f"Processing time per frame: {round(end_time-start_time,2)} s")
            else:
                print("Empty camera buffer...")
                # return



        










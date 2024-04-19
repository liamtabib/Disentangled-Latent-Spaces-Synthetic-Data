import bz2
import os

import cv2
import dlib
import numpy as np
import wget


class LandmarkDetector:
    def __init__(self, pretrained_model_root="./pretrained_models"):
        super(LandmarkDetector, self).__init__()
        os.makedirs(pretrained_model_root, exist_ok=True)

        self.pretrained_model_root = pretrained_model_root
        self.cnn_model_file = "mmod_human_face_detector.dat"
        self.landmark_prediction_file = "shape_predictor_68_face_landmarks.dat"

        for file in [self.cnn_model_file, self.landmark_prediction_file]:
            if not os.path.exists(f"{pretrained_model_root}/{file}"):
                self.download_pretrained_model(file)

    def detect_facial_landmark(self, img, detector_type="HOG"):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the face with HOG+Linear SVM
        if detector_type == "HOG":
            detected_face = self.get_HOG_detector(img)
            if len(detected_face) == 0:
                detected_face = self.get_CNN_detector(img_gray)
                rect = dlib.rectangle(
                    detected_face[0],
                    detected_face[1],
                    detected_face[2],
                    detected_face[3],
                )

            else:
                rect = detected_face[0]

        # If HOG+Linear SVM face detector fails to detect a face, try CNN face detector
        elif detector_type == "CNN":
            detected_face = self.get_CNN_detector(img_gray)
            rect = dlib.rectangle(
                detected_face[0], detected_face[1], detected_face[2], detected_face[3]
            )

        # Predict the facial landmark
        predictor = dlib.shape_predictor(
            f"{self.pretrained_model_root}/{self.landmark_prediction_file}"
        )
        predicted_landmark = predictor(img_gray, rect)

        src_landmark = np.zeros((68, 2), dtype="float")
        for i in range(68):
            src_landmark[i] = (
                predicted_landmark.part(i).x,
                predicted_landmark.part(i).y,
            )

        return src_landmark

    @staticmethod
    def get_HOG_detector(img):
        detector = dlib.get_frontal_face_detector()
        detected_face = detector(img, 1)

        return detected_face

    def get_CNN_detector(self, img):
        cnn_face_detector = dlib.cnn_face_detection_model_v1(
            f"{self.pretrain_model_root}/{self.cnn_model_file}"
        )
        faces_cnn = cnn_face_detector(img, 1)
        for face in faces_cnn:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right()
            h = face.rect.bottom()

        return x, y, w, h

    def download_pretrained_model(self, file_name):
        # Download file
        base_url = "http://dlib.net/files"
        url = base_url + "/" + file_name + ".bz2"
        wget.download(url, self.pretrained_model_root)

        with bz2.open(f"{self.pretrained_model_root}/{file_name + '.bz2'}", "rb") as f:
            uncompressed_file = f.read()

        # store decompressed file
        with open(f"{self.pretrained_model_root}/{file_name}", "wb") as f:
            f.write(uncompressed_file)
            f.close()

        os.remove(f"{self.pretrained_model_root}/{file_name + '.bz2'}")

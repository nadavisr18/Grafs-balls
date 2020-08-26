import cv2
import time
import numpy as np


class Processor:
    def __init__(self):
        self.vid = cv2.VideoCapture("video.mp4")
        self.frame = 0

    def __del__(self):
        self.vid.release()
        cv2.destroyAllWindows()

    def display_video(self):
        while True:
            ret, frame = self.vid.read()
            if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
                break
            frame = self.get_ball(frame)
            cv2.imshow('frame', frame)
            # time.sleep(0.016)

    def get_ball(self, frame):
        small = cv2.resize(frame, (0, 0), fx=0.02, fy=0.02)
        only_ball = np.apply_along_axis(self.threshold, 2, small)
        big = cv2.resize(only_ball, (0, 0), fx=30, fy=30)
        return big

    @staticmethod
    def threshold(x):
        dist = abs(np.linalg.norm(np.array([125, 234, 255]) - x))
        if dist < 40:
            return x
        else:
            return x//100

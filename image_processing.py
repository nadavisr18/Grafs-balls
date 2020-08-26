import cv2
import time
import numpy as np


class Processor:
    def __init__(self):
        self.vid = cv2.VideoCapture("video.mp4")
        self.downscale_factor = 50
        self.ball_radius = 100
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
            time.sleep(0.016)

    def get_ball(self, frame):
        small = cv2.resize(frame, (0, 0), fx=1 / self.downscale_factor, fy=1 / self.downscale_factor)

        only_ball = np.apply_along_axis(self.threshold, 2, small)
        ball_indexes = np.where(only_ball == True)

        ball_x = np.average(ball_indexes[0])
        ball_y = np.average(ball_indexes[1])

        from_x, to_x, from_y, to_y = self.get_ball_range(ball_x, ball_y, frame.shape[:2])
        ball_in_frame = frame[from_x:to_x, from_y:to_y]
        return ball_in_frame

    def get_ball_range(self, x, y, frame_size):
        if str(x) == "nan":
            return 0, 1, 0, 1
        upscaled_x = np.floor(x * self.downscale_factor)
        min_x = max(0, upscaled_x - self.ball_radius)
        max_x = min(frame_size[0], upscaled_x + self.ball_radius)

        upscaled_y = np.floor(y * self.downscale_factor)
        min_y = max(0, upscaled_y - self.ball_radius)
        max_y = min(frame_size[1], upscaled_y + self.ball_radius)
        return int(min_x), int(max_x), int(min_y), int(max_y)

    @staticmethod
    def threshold(x):
        dist = abs(np.linalg.norm(np.array([125, 234, 255]) - x))
        if dist < 40:
            return True
        else:
            return False

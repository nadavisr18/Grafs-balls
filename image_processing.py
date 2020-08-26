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
            time.sleep(0.1)

    def get_ball(self, frame):
        small = cv2.resize(frame, (0, 0), fx=1 / self.downscale_factor, fy=1 / self.downscale_factor)

        only_ball = np.apply_along_axis(self.threshold, 2, small)
        ball_indexes = np.where(only_ball == True)

        ball_x = np.average(ball_indexes[0])
        ball_y = np.average(ball_indexes[1])

        from_x, to_x, from_y, to_y = self.get_ball_range(ball_x, ball_y, frame)
        ball_in_frame = frame[from_x:to_x, from_y:to_y]
        if 0 in ball_in_frame.shape:
            ball_in_frame = self.handle_no_ball()
        return ball_in_frame

    def get_ball_range(self, x, y, frame):
        if str(x) == "nan":
            return 0, 1, 0, 1
        upscaled_x = int(np.floor(x * self.downscale_factor))
        upscaled_y = int(np.floor(y * self.downscale_factor))
        min_x = self.walk(frame, upscaled_x, upscaled_y, [-1,0])[0] - 10
        max_x = self.walk(frame, upscaled_x, upscaled_y, [ 1,0])[0] + 10
        min_y = self.walk(frame, upscaled_x, upscaled_y, [0,-1])[1] - 10
        max_y = self.walk(frame, upscaled_x, upscaled_y, [0, 1])[1] + 10

        return int(min_x), int(max_x), int(min_y), int(max_y)

    def walk(self, frame, x, y,direction):
        while True:
            x += direction[0]
            y += direction[1]
            pixel = frame[x,y]
            if not self.threshold(pixel):
                return [x,y]

    @staticmethod
    def handle_no_ball():
        return np.ones((2,2,3))

    @staticmethod
    def threshold(x):
        dist = abs(np.linalg.norm(np.array([125, 234, 255]) - x))
        if dist < 40:
            return True
        else:
            return False
